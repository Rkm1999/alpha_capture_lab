#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface ACLPanoramaStitcher : NSObject

+ (nullable NSData *)stitchFrames:(NSArray<NSData *> *)frames
                          preview:(BOOL)preview
                            error:(NSError * _Nullable * _Nullable)error;

@end

NS_ASSUME_NONNULL_END
