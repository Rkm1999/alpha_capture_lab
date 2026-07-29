plugins {
    id("com.android.application")
}

android {
    namespace = "com.ryu.scunetdenoiser"
    compileSdk = 36

    defaultConfig {
        applicationId = "com.ryu.scunetdenoiser"
        minSdk = 29
        targetSdk = 36
        versionCode = 7
        versionName = "0.1.0-beta.5"

        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    sourceSets {
        getByName("main") {
            java.srcDir("../runtime-android/src/main/java")
            assets.srcDir("../runtime-android/src/main/assets")
            jniLibs.srcDir("../runtime-android/src/main/jniLibs")
        }
    }

    androidResources {
        noCompress += "tflite"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    lint {
        // Qualcomm's bundled compiler and HTP runtime are arm64-only.
        disable += "ChromeOsAbiSupport"
    }
}

dependencies {
    implementation("androidx.exifinterface:exifinterface:1.3.7")
    implementation("com.google.ai.edge.litert:litert:2.1.6")
    testImplementation("junit:junit:4.13.2")
}
