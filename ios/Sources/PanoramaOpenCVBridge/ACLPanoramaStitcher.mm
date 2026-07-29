#import "ACLPanoramaStitcher.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace {

constexpr int kFrameEdge = 2048;
constexpr int kPreviewEdge = 1600;
constexpr int kFeatures = 5000;
constexpr float kRatio = 0.75f;
constexpr int kMinMatches = 18;
constexpr int kMinInliers = 12;
constexpr double kMinInlierRatio = 0.3;
constexpr double kRansacThreshold = 3.0;
constexpr int kRansacIterations = 2500;
constexpr double kRansacConfidence = 0.995;
constexpr double kMinScale = 0.25;
constexpr double kMaxScale = 4.0;
constexpr int kPreviewMaxEdge = 6000;
constexpr int64_t kPreviewMaxPixels = 6'000'000;
constexpr int64_t kFinalMaxPixels = 60'000'000;
constexpr float kMinBlendWeight = 0.001f;
constexpr float kMinFeatherWeight = 0.02f;

cv::Mat decode(NSData *data, int maxEdge) {
    std::vector<uint8_t> bytes(data.length);
    [data getBytes:bytes.data() length:data.length];
    cv::Mat image = cv::imdecode(bytes, cv::IMREAD_COLOR);
    if (image.empty()) throw std::runtime_error("A panorama frame could not be decoded");
    const int longest = std::max(image.cols, image.rows);
    if (maxEdge > 0 && longest > maxEdge) {
        const double scale = static_cast<double>(maxEdge) / longest;
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(), scale, scale, cv::INTER_AREA);
        return resized;
    }
    return image;
}

double polygonArea(const std::vector<cv::Point2f> &points) {
    double twiceArea = 0;
    for (size_t index = 0; index < points.size(); ++index) {
        const auto &next = points[(index + 1) % points.size()];
        twiceArea += points[index].x * next.y - next.x * points[index].y;
    }
    return std::abs(twiceArea) / 2;
}

cv::Mat findHomography(const cv::Mat &source, const cv::Mat &destination) {
    cv::Mat sourceGray;
    cv::Mat destinationGray;
    cv::cvtColor(source, sourceGray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(destination, destinationGray, cv::COLOR_BGR2GRAY);

    auto orb = cv::ORB::create(kFeatures);
    std::vector<cv::KeyPoint> sourceKeys;
    std::vector<cv::KeyPoint> destinationKeys;
    cv::Mat sourceDescriptors;
    cv::Mat destinationDescriptors;
    orb->detectAndCompute(sourceGray, cv::noArray(), sourceKeys, sourceDescriptors);
    orb->detectAndCompute(
        destinationGray,
        cv::noArray(),
        destinationKeys,
        destinationDescriptors
    );
    if (sourceDescriptors.empty() || destinationDescriptors.empty()) {
        throw std::runtime_error("This frame has too little detail to align");
    }

    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> pairs;
    matcher.knnMatch(sourceDescriptors, destinationDescriptors, pairs, 2);
    std::vector<cv::DMatch> accepted;
    for (const auto &pair : pairs) {
        if (pair.size() >= 2 && pair[0].distance < kRatio * pair[1].distance) {
            accepted.push_back(pair[0]);
        }
    }
    if (accepted.size() < kMinMatches) {
        throw std::runtime_error("Overlap this frame more with the previous image");
    }

    std::vector<cv::Point2f> sourcePoints;
    std::vector<cv::Point2f> destinationPoints;
    for (const auto &match : accepted) {
        sourcePoints.push_back(sourceKeys[match.queryIdx].pt);
        destinationPoints.push_back(destinationKeys[match.trainIdx].pt);
    }
    cv::Mat inlierMask;
    cv::Mat homography = cv::findHomography(
        sourcePoints,
        destinationPoints,
        cv::RANSAC,
        kRansacThreshold,
        inlierMask,
        kRansacIterations,
        kRansacConfidence
    );
    if (homography.empty()) {
        throw std::runtime_error("The panorama frame could not be aligned");
    }
    const int inliers = cv::countNonZero(inlierMask);
    if (inliers < kMinInliers ||
        static_cast<double>(inliers) / accepted.size() < kMinInlierRatio) {
        throw std::runtime_error("Overlap this frame more with the previous image");
    }

    std::vector<cv::Point2f> corners = {
        {0, 0},
        {static_cast<float>(source.cols), 0},
        {static_cast<float>(source.cols), static_cast<float>(source.rows)},
        {0, static_cast<float>(source.rows)},
    };
    std::vector<cv::Point2f> projected;
    cv::perspectiveTransform(corners, projected, homography);
    for (const auto &point : projected) {
        if (!std::isfinite(point.x) || !std::isfinite(point.y)) {
            throw std::runtime_error("The panorama transform was unstable");
        }
    }
    const double scale = polygonArea(projected) /
        static_cast<double>(source.cols * source.rows);
    if (scale < kMinScale || scale > kMaxScale) {
        throw std::runtime_error("Move more steadily between panorama frames");
    }
    return homography;
}

std::vector<cv::Mat> transformsFor(const std::vector<cv::Mat> &images) {
    std::vector<cv::Mat> transforms;
    transforms.push_back(cv::Mat::eye(3, 3, CV_64F));
    for (size_t index = 1; index < images.size(); ++index) {
        cv::Mat pair = findHomography(images[index], images[index - 1]);
        transforms.push_back(transforms.back() * pair);
    }
    return transforms;
}

struct Bounds {
    double minX;
    double minY;
    double maxX;
    double maxY;
};

Bounds boundsFor(
    const std::vector<cv::Mat> &images,
    const std::vector<cv::Mat> &transforms
) {
    Bounds bounds{
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
        -std::numeric_limits<double>::infinity(),
    };
    for (size_t index = 0; index < images.size(); ++index) {
        std::vector<cv::Point2f> corners = {
            {0, 0},
            {static_cast<float>(images[index].cols), 0},
            {
                static_cast<float>(images[index].cols),
                static_cast<float>(images[index].rows),
            },
            {0, static_cast<float>(images[index].rows)},
        };
        std::vector<cv::Point2f> projected;
        cv::perspectiveTransform(corners, projected, transforms[index]);
        for (const auto &point : projected) {
            bounds.minX = std::min(bounds.minX, static_cast<double>(point.x));
            bounds.minY = std::min(bounds.minY, static_cast<double>(point.y));
            bounds.maxX = std::max(bounds.maxX, static_cast<double>(point.x));
            bounds.maxY = std::max(bounds.maxY, static_cast<double>(point.y));
        }
    }
    if (!std::isfinite(bounds.minX) || !std::isfinite(bounds.minY) ||
        !std::isfinite(bounds.maxX) || !std::isfinite(bounds.maxY)) {
        throw std::runtime_error("The panorama bounds were invalid");
    }
    return {
        std::floor(bounds.minX),
        std::floor(bounds.minY),
        std::ceil(bounds.maxX),
        std::ceil(bounds.maxY),
    };
}

cv::Mat featherWeight(const cv::Size &size) {
    cv::Mat mask(size, CV_8UC1, cv::Scalar(255));
    mask.row(0).setTo(0);
    mask.row(mask.rows - 1).setTo(0);
    mask.col(0).setTo(0);
    mask.col(mask.cols - 1).setTo(0);
    cv::Mat distance;
    cv::distanceTransform(mask, distance, cv::DIST_L2, 3);
    cv::Mat normalized;
    cv::normalize(
        distance,
        normalized,
        kMinFeatherWeight,
        1.0,
        cv::NORM_MINMAX
    );
    return normalized;
}

cv::Mat renderPreview(
    const std::vector<cv::Mat> &images,
    const std::vector<cv::Mat> &transforms
) {
    const Bounds bounds = boundsFor(images, transforms);
    const double rawWidth = std::max(1.0, bounds.maxX - bounds.minX);
    const double rawHeight = std::max(1.0, bounds.maxY - bounds.minY);
    const double edgeScale = std::min({
        static_cast<double>(kPreviewMaxEdge) / rawWidth,
        static_cast<double>(kPreviewMaxEdge) / rawHeight,
        1.0,
    });
    const double pixelScale = std::min(
        std::sqrt(kPreviewMaxPixels / (rawWidth * rawHeight)),
        1.0
    );
    const double scale = std::min(edgeScale, pixelScale);
    const int width = std::max(1, static_cast<int>(std::ceil(rawWidth * scale)));
    const int height = std::max(1, static_cast<int>(std::ceil(rawHeight * scale)));
    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 0) = scale;
    translation.at<double>(1, 1) = scale;
    translation.at<double>(0, 2) = -bounds.minX * scale;
    translation.at<double>(1, 2) = -bounds.minY * scale;

    cv::Mat weightedSum = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat weightSum = cv::Mat::zeros(height, width, CV_32FC1);
    for (size_t index = 0; index < images.size(); ++index) {
        const cv::Mat canvasTransform = translation * transforms[index];
        cv::Mat warpedImage = cv::Mat::zeros(height, width, CV_8UC3);
        cv::Mat warpedWeight = cv::Mat::zeros(height, width, CV_32FC1);
        cv::warpPerspective(
            images[index],
            warpedImage,
            canvasTransform,
            {width, height},
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT
        );
        cv::Mat sourceWeight = featherWeight(images[index].size());
        cv::warpPerspective(
            sourceWeight,
            warpedWeight,
            canvasTransform,
            {width, height},
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT
        );
        cv::Mat imageFloat;
        warpedImage.convertTo(imageFloat, CV_32FC3);
        std::vector<cv::Mat> weights = {warpedWeight, warpedWeight, warpedWeight};
        cv::Mat weight3;
        cv::merge(weights, weight3);
        cv::multiply(imageFloat, weight3, imageFloat);
        weightedSum += imageFloat;
        weightSum += warpedWeight;
    }
    cv::max(weightSum, kMinBlendWeight, weightSum);
    std::vector<cv::Mat> channels;
    cv::split(weightedSum, channels);
    for (auto &channel : channels) cv::divide(channel, weightSum, channel);
    cv::merge(channels, weightedSum);
    cv::Mat output;
    weightedSum.convertTo(output, CV_8UC3);
    return output;
}

cv::Mat renderFinal(
    const std::vector<cv::Mat> &images,
    const std::vector<cv::Mat> &transforms
) {
    const Bounds bounds = boundsFor(images, transforms);
    const double rawWidth = std::max(1.0, bounds.maxX - bounds.minX);
    const double rawHeight = std::max(1.0, bounds.maxY - bounds.minY);
    const double scale = std::min(
        std::sqrt(kFinalMaxPixels / (rawWidth * rawHeight)),
        1.0
    );
    const int width = std::max(1, static_cast<int>(std::floor(rawWidth * scale)));
    const int height = std::max(1, static_cast<int>(std::floor(rawHeight * scale)));
    cv::Mat translation = cv::Mat::eye(3, 3, CV_64F);
    translation.at<double>(0, 0) = scale;
    translation.at<double>(1, 1) = scale;
    translation.at<double>(0, 2) = -bounds.minX * scale;
    translation.at<double>(1, 2) = -bounds.minY * scale;

    cv::Mat output = cv::Mat::zeros(height, width, CV_8UC3);
    cv::Mat sourceMask;
    for (size_t index = 0; index < images.size(); ++index) {
        cv::Mat warped;
        cv::Mat mask(images[index].size(), CV_8UC1, cv::Scalar(255));
        const cv::Mat canvasTransform = translation * transforms[index];
        cv::warpPerspective(
            images[index],
            warped,
            canvasTransform,
            {width, height},
            cv::INTER_LINEAR,
            cv::BORDER_CONSTANT
        );
        cv::warpPerspective(
            mask,
            sourceMask,
            canvasTransform,
            {width, height},
            cv::INTER_NEAREST,
            cv::BORDER_CONSTANT
        );
        warped.copyTo(output, sourceMask);
    }
    return output;
}

std::vector<uchar> encode(const cv::Mat &image) {
    std::vector<uchar> output;
    if (!cv::imencode(".jpg", image, output, {cv::IMWRITE_JPEG_QUALITY, 95})) {
        throw std::runtime_error("The panorama could not be encoded");
    }
    return output;
}

}  // namespace

@implementation ACLPanoramaStitcher

+ (NSData *)stitchFrames:(NSArray<NSData *> *)frames
                 preview:(BOOL)preview
                   error:(NSError **)error {
    try {
        if (frames.count == 0) {
            throw std::runtime_error("No panorama frames were captured");
        }
        std::vector<cv::Mat> registrationImages;
        for (NSData *data in frames) {
            registrationImages.push_back(decode(data, kFrameEdge));
        }
        const auto registrationTransforms = transformsFor(registrationImages);
        cv::Mat result;
        if (preview) {
            result = renderPreview(registrationImages, registrationTransforms);
            const int longest = std::max(result.cols, result.rows);
            if (longest > kPreviewEdge) {
                const double scale = static_cast<double>(kPreviewEdge) / longest;
                cv::resize(result, result, cv::Size(), scale, scale, cv::INTER_AREA);
            }
        } else {
            std::vector<cv::Mat> originals;
            std::vector<cv::Mat> fullTransforms;
            for (NSData *data in frames) {
                originals.push_back(decode(data, 0));
            }
            const cv::Mat referenceScale = (cv::Mat_<double>(3, 3) <<
                static_cast<double>(originals[0].cols) /
                    registrationImages[0].cols, 0, 0,
                0, static_cast<double>(originals[0].rows) /
                    registrationImages[0].rows, 0,
                0, 0, 1);
            for (NSUInteger index = 0; index < frames.count; ++index) {
                const cv::Mat sourceScale = (cv::Mat_<double>(3, 3) <<
                    static_cast<double>(registrationImages[index].cols) /
                        originals[index].cols, 0, 0,
                    0, static_cast<double>(registrationImages[index].rows) /
                        originals[index].rows, 0,
                    0, 0, 1);
                fullTransforms.push_back(
                    referenceScale * registrationTransforms[index] * sourceScale
                );
            }
            result = renderFinal(originals, fullTransforms);
        }
        const auto jpeg = encode(result);
        return [NSData dataWithBytes:jpeg.data() length:jpeg.size()];
    } catch (const std::exception &exception) {
        if (error) {
            *error = [NSError errorWithDomain:@"AlphaCaptureLab.Panorama"
                                         code:1
                                     userInfo:@{
                NSLocalizedDescriptionKey: @(exception.what())
            }];
        }
        return nil;
    }
}

@end
