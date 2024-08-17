[app]

# (str) Title of your application
title = My Application

# (str) Package name
package.name = myapp

# (str) Package domain (needed for android/ios packaging)
package.domain = org.test

# (str) Source code where the main.py lives
source.dir = .

# (list) Source files to include (let empty to include all the files)
source.include_exts = py,png,jpg,kv,atlas

# (list) Source files to exclude (let empty to not exclude anything)
# source.exclude_exts = spec

# (list) List of directories to exclude (let empty to not exclude anything)
# source.exclude_dirs = tests, bin, venv

# (list) List of exclusions using pattern matching
# source.exclude_patterns = license,images/*/*.jpg

# (str) Application versioning (method 1)
version = 0.1

# (list) Application requirements
requirements = python3,kivy

# (str) Custom source folders for requirements
# requirements.source.kivy = ../../kivy

# (str) Presplash of the application
# presplash.filename = %(source.dir)s/data/presplash.png

# (str) Icon of the application
# icon.filename = %(source.dir)s/data/icon.png

# (list) Supported orientations
orientation = portrait

# (list) List of services to declare
# services = NAME:ENTRYPOINT_TO_PY,NAME2:ENTRYPOINT2_TO_PY

#
# OSX Specific
#

# (str) author = © Copyright Info

# (str) change the major version of python used by the app
osx.python_version = 3

# (str) Kivy version to use
osx.kivy_version = 1.9.1

#
# Android specific
#

# (bool) Indicate if the application should be fullscreen or not
fullscreen = 0

# (string) Presplash background color (for android toolchain)
# android.presplash_color = #FFFFFF

# (string) Presplash animation using Lottie format.
# android.presplash_lottie = "path/to/lottie/file.json"

# (str) Adaptive icon of the application (used if Android API level is 26+ at runtime)
# icon.adaptive_foreground.filename = %(source.dir)s/data/icon_fg.png
# icon.adaptive_background.filename = %(source.dir)s/data/icon_bg.png

# (list) Permissions
# android.permissions = android.permission.INTERNET, (name=android.permission.WRITE_EXTERNAL_STORAGE;maxSdkVersion=18)

# (list) Features (adds uses-feature -tags to manifest)
# android.features = android.hardware.usb.host

# (int) Target Android API, should be as high as possible.
# android.api = 31

# (int) Minimum API your APK / AAB will support.
# android.minapi = 21

# (int) Android SDK version to use
# android.sdk = 20

# (str) Android NDK version to use
# android.ndk = 23b

# (int) Android NDK API to use. This is the minimum API your app will support, it should usually match android.minapi.
# android.ndk_api = 21

# (bool) Use --private data storage (True) or --dir public storage (False)
# android.private_storage = True

# (str) Android NDK directory (if empty, it will be automatically downloaded.)
# android.ndk_path =

# (str) Android SDK directory (if empty, it will be automatically downloaded.)
# android.sdk_path =

# (str) ANT directory (if empty, it will be automatically downloaded.)
# android.ant_path =

# (bool) If True, then skip trying to update the Android sdk
# android.skip_update = False

# (bool) If True, then automatically accept SDK license agreements. This is intended for automation only. If set to False, the default, you will be shown the license when first running buildozer.
# android.accept_sdk_license = False

# (str) Android entry point, default is ok for Kivy-based app
# android.entrypoint = org.kivy.android.PythonActivity

# (str) Full name including package path of the Java class that implements Android Activity
# android.activity_class_name = org.kivy.android.PythonActivity

# (str) Extra xml to write directly inside the <manifest> element of AndroidManifest.xml
# android.extra_manifest_xml = ./src/android/extra_manifest.xml

# (str) Extra xml to write directly inside the <manifest><application> tag of AndroidManifest.xml
# android.extra_manifest_application_arguments = ./src/android/extra_manifest_application_arguments.xml

# (str) Full name including package path of the Java class that implements Python Service
# android.service_class_name = org.kivy.android.PythonService

# (str) Android app theme, default is ok for Kivy-based app
# android.apptheme = "@android:style/Theme.NoTitleBar"

# (list) Pattern to whitelist for the whole project
# android.whitelist =

# (str) Path to a custom whitelist file
# android.whitelist_src =

# (str) Path to a custom blacklist file
# android.blacklist_src =

# (list) List of Java .jar files to add to the libs so that pyjnius can access their classes. Don't add jars that you do not need, since extra jars can slow down the build process. Allows wildcards matching, for example: OUYA-ODK/libs/*.jar
# android.add_jars = foo.jar,bar.jar,path/to/more/*.jar

# (list) List of Java files to add to the android project (can be java or a directory containing the files)
# android.add_src =

# (list) Android AAR archives to add
# android.add_aars =

# (list) Put these files or directories in the apk assets directory.
# android.add_assets = source_asset_relative_path
# android.add_assets = source_asset_path:destination_asset_relative_path
# android.add_assets =

# (list) Put these files or directories in the apk res directory.
# android.add_resources = my_icons/all-inclusive.png:drawable/all_inclusive.png
# android.add_resources = legal_icons:drawable
# android.add_resources = legal_resources
# android.add_resources =

# (list) Gradle dependencies to add
# android.gradle_dependencies =

# (bool) Enable AndroidX support. Enable when 'android.gradle_dependencies' contains an 'androidx' package, or any package from Kotlin source.
# android.enable_androidx = True

# (list) add java compile options
# android.add_compile_options = "sourceCompatibility = 1.8", "targetCompatibility = 1.8"

# (list) Gradle repositories to add {can be necessary for some android.gradle_dependencies}
# android.add_gradle_repositories = "maven { url 'https://kotlin.bintray.com/ktor' }"

# (list) packaging options to add 
# android.add_packaging_options = "exclude 'META-INF/common.kotlin_module'", "exclude 'META-INF/*.kotlin_module'"

# (list) Java classes to add as activities to the manifest.
# android.add_activities = com.example.ExampleActivity

# (str) OUYA Console category. Should be one of GAME or APP
# android.ouya.category = GAME

# (str) Filename of OUYA Console icon. It must be a 732x412 png image.
# android.ouya.icon.filename = %(source.dir)s/data/ouya_icon.png

# (str) XML file to include as an intent filters in <activity> tag
# android.manifest.intent_filters =

# (list) Copy these files to src/main/res/xml/ (used for example with intent-filters)
# android.res_xml = PATH_TO_FILE,

# (str) launchMode to set for the main activity
# android.manifest.launch_mode = standard

# (str) screenOrientation to set for the main activity.
# Valid values can be found at https://developer.android.com/guide/topics/manifest/activity-element
# android.manifest.orientation = fullSensor

# (list) Android additional libraries to copy into libs/armeabi
# android.add_libs_armeabi = libs/android/*.so
# android.add_libs_armeabi_v7a = libs/android-v7/*.so
# android.add_libs_arm64_v8a = libs/android-v8/*.so
# android.add_libs_x86 = libs/android-x86/*.so
# android.add_libs_mips = libs/android-mips/*.so

# (bool) Indicate whether the screen should stay on
# android.wakelock = False

# (list) Android application meta-data to set (key=value format)
# android.meta_data =

# (list) Android library project to add (will be added in the project.properties automatically.)
# android.library_references =

# (list) Android shared libraries which will be added to AndroidManifest.xml using <uses-library> tag
# android.uses_library =

# (str) Android logcat filters to use
# android.logcat_filters = *:S python:D

# (bool) Android logcat only display log for activity's pid
# android.logcat_pid_only = False

# (str) Android additional adb arguments
# android.adb_args = -H host.docker.internal

# (bool) Copy library instead of making a libpymodules.so
# android.copy_libs = 1

# (list) The Android archs to build for, choices: armeabi-v7a, arm64-v8a, x86, x86_64
android.arch = armeabi-v7a,arm64-v8a

# (bool) whether to use the Python 3 version
python3 = True

#
# iOS specific
#

# (bool) Whether to automatically manage your ios app's signing
# ios.codesigning = False

# (bool) Enable automatic code signing
# ios.codesigning = True

# (str) Application certificate to use
# ios.codesigning.identity = iPhone Developer: John Doe (ABCDEFGHIJ)

# (str) Development team ID
# ios.development_team = ABCD1234

# (bool) Whether to enable testflight
# ios.testflight = False

# (str) Release version (can be empty)
# ios.version = 1.0.0

# (bool) Whether to use Xcode 14.0 or later
# ios.xcode_version = 14.0

# (str) Development team ID (for iOS development)
# ios.development_team = YOUR_DEVELOPMENT_TEAM_ID

# (bool) Enable automatic code signing
# ios.codesigning = True

# (bool) Whether to use xcodebuild instead of Xcode IDE
# ios.xcodebuild = False

# (str) Signing certificate to use
# ios.codesigning.identity = iPhone Developer: John Doe (ABCDEFGHIJ)

# (str) path to provisioning profile file
# ios.codesigning.provisioning_profile_path = path/to/provisioning_profile.mobileprovision

# (str) path to the app-specific development provisioning profile file
# ios.codesigning.app_provisioning_profile_path = path/to/app_provisioning_profile.mobileprovision

# (list) Extra flags to pass to xcodebuild
# ios.xcodebuild_flags =

# (bool) Whether to enable verbose logging
# ios.verbose = False
