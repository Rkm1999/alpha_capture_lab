# Real-Device Test Checklist

## Matrix

Run the core flow on physical Android 10, 13, and 16 devices where available.
Record the A6300 firmware version, camera remote-app name/version, lens, and
whether `Original` appears in `getAvailablePostviewImageSize`.

## Connection And Recovery

- Connect with cellular data enabled and confirm discovery uses camera Wi-Fi.
- Deny Android 16 Nearby devices once, retry, then grant it.
- Turn the camera off during discovery, live view, capture, and download.
- Let the camera sleep, wake it, and reconnect without force-closing Android.
- Background/foreground the app and rotate during live view.
- Switch away from the camera Wi-Fi and confirm immediate disconnected state.

## Camera Control

- Confirm every visible setting is present in the advertised API list.
- Change capture type/mode and verify dependent controls refresh.
- Test auto and manual exposure, a long exposure, and a camera-busy response.
- Capture 20 consecutive JPEGs without restarting the app.
- Confirm `Original` is selected only when offered; otherwise confirm the 2 MP warning.
- With the app connected and untouched, press the camera-body shutter twice.
  Confirm two camera-card exposures produce exactly two new phone JPEGs and two
  filmstrip items, with no duplicate exposure or transfer.
- Press the camera-body shutter again while the preceding JPEG is still importing.
  Confirm event polling continues, `IMPORTING` stays visible while work is pending,
  and both JPEGs are eventually saved in capture order without a duplicate.
- Set Drive to Continuous, hold and release the camera-body shutter, and confirm
  every v1.2 continuous postview becomes one ordered phone JPEG.
- Hold the app shutter for one second and release it. Confirm stop is requested
  and all URLs import. Repeat past five seconds and confirm the watchdog stops it.

## Computational Capture

- Capture a full-resolution photo, open it from the filmstrip, preview every
  LUT, save a derivative, and verify dimensions and EXIF.
- Run Live ND at 2, 4, 8, and 16 frames. Confirm every accepted frame appears
  in the filmstrip, the preview advances, the final is saved, and `Original`
  postview is restored.
- Repeat Live ND with Burst selected. Confirm short batches repeat to the target,
  extras remain visible, the prior Drive mode is restored, and Single still works.
- Start Live Composite, introduce a brighter moving light after the base frame,
  stop during an exposure, and confirm the current exposure finishes before save.
- Select Panorama and tap Start once. Confirm the app changes Drive to `Single`
  and requests `Original` when available, then use only the camera-body shutter to capture
  frames individually with 30-50% overlap. Confirm `IMPORTING` clears and the
  `PANO` count advances after each accepted frame, reject one low-overlap frame,
  then tap Finish on the phone and inspect the expanded JPEG for seams.
- Background during ND/Composite and confirm the current capture finishes, the
  partial result is saved when at least two frames exist, and live view resumes.
- Cancel an app-shutter Panorama and confirm no final stitched JPEG appears in the
  Gallery. Repeat with body-shutter source frames and confirm those already-imported
  source JPEGs remain while no final stitched JPEG is added.

## S25 / A6300 Verification (2026-07-10)

- Device: Samsung SM-S931W, Android 16; camera: ILCE-6300 with Smart Remote
  Control SR/4.31, API 2.1.4.
- Safe 2 MP Single cadence measured at 2.985 s exposure/RPC plus 1.274 s for a
  443,762-byte 1616x1080 transfer.
- Full 6000x4000 Photo and 3D Cinema LUT derivative saved successfully.
- Four-frame Live ND saved a 1616x1080 progressive result in about 22 s.
- Live Composite stopped after three accepted frames and saved a 1616x1080
  EXIF-tagged result in about 18 s.
- Two physical stationary panorama frames registered live and saved a
  1618x1082 EXIF-tagged result. A connected S25 OpenCV test with known
  overlapping crops separately verifies expansion into a wider mosaic.
- Connected device tests verify ND averaging, brighter-only Composite pixels,
  panorama expansion, and OpenCV initialization.
- The final debug APK was installed after the state-cleanup and panorama-memory
  changes. A fresh two-frame Live ND saved
  `SONY_LIVE_ND_20260710_213928_874.jpg` at 1616x1080 with A6300 EXIF.
- A final two-frame Live Composite was stopped during its second exposure; it
  finished that frame and saved `SONY_COMPOSITE_20260710_214348_262.jpg` at
  1616x1080 with A6300 EXIF.
- A final two-frame panorama registered both frames in the live mosaic and
  saved `SONY_PANORAMA_20260710_214304_629.jpg` at 1618x1081 with A6300 EXIF.
  The physical camera was stationary; the connected synthetic-overlap test is
  the evidence for a genuinely widened mosaic.
- A new 6000x4000 original capture produced a 6000x4000 Cinema LUT derivative,
  `SONY_LUT_20260710_214149_654.jpg`, with `SONY / ILCE-6300` EXIF preserved.
- Live ND, Live Composite, completed panorama, and discarded panorama all left
  `getAvailablePostviewImageSize` reporting `Original` as the current size.
- Final gates: 43 JVM tests, two connected S25 instrumentation tests, Android
  lint with zero errors, debug assembly, and minified release assembly passed.
- Movie capture was not tested because it is intentionally outside the current
  product scope and is not exposed in the UI.

## Preview-First And Contents Transfer Verification (2026-07-11)

- An app-triggered Single shot first created an explicit `Live view` placeholder;
  no captured JPEG was implied and the placeholder could not enter the LUT editor.
- A physical 2M capture imported a 160x120 preview first and exposed **Import
  original** only because the A6300 advertised its `avContent` endpoint and
  `Contents Transfer` camera function.
- Importing that Original switched camera function temporarily, matched the
  timestamped content record, and saved a 6000x4000 JPEG. The preview remained
  in the capture record and the displayed asset changed to Original.
- Direct service verification returned `DSC00001.JPG` from `getContentList` v1.3;
  the app restored `Remote Shooting`, capabilities, event monitoring, and live view.
- The Original queue has independent queued/downloading/failed states, retry,
  cancellation, request deduplication, JPEG dimension validation, and pauses for
  active computational or burst work.

## 2026-07-11 Implementation Gates

- 82 JVM tests and two connected Samsung S25 instrumentation tests passed.
- Android lint completed without errors; debug and minified release assemblies passed.
- Automated coverage includes completed burst batches, rolling burst calibration,
  exact accepted Live ND targets, LUT intensity/state, preview routing, photo-control
  priority, Contents Transfer parsing, panorama resource planning, and cleanup gates.

## Final Phase Verification (2026-07-11)

- The latest build recovered a decoded live view after a fresh install/connect and
  showed the read-only A6300 mode prominently as `A / Aperture`.
- Selecting Warm applied the repeated-frame live preview without blocking controls.
  With Save LUT enabled, a physical/app burst retained its unprocessed Preview and
  created separate `Warm Preview` derivatives. Earlier Original-path verification
  produced a separate 6000x4000 LUT derivative from an unchanged 6000x4000 source.
- Calibrated Live ND completed and saved `SONY_LIVE_ND_20260711_155035_067.jpg`
  at 1616x1080. The run retained all source JPEGs and restored Single fallback
  correctly when the A6300's burst API became transiently unavailable.
- A two-frame panorama kept the stitched result in the main viewer while the live
  camera feed remained independently visible in the lower-right PIP. Both source
  files remained as 6000x4000 `SONY_PANO_SOURCE` JPEGs.
- Finish rendered `SONY_PANORAMA_20260711_163224_828.jpg` at 3591x2396 from those
  Originals. The S25 heap budget required a 0.60-scale final, so it was honestly
  labeled `Panorama Preview`; this is substantially above the 1616x1080 alignment
  preview and did not require holding both decoded Originals in memory.

## Physical Shutter Verification (2026-07-10)

- Installed the final debug APK on the Samsung SM-S931W and connected to the
  A6300 without touching the phone after listener readiness.
- Before the test, the phone contained 35 Sony Remote JPEGs and the camera
  reported 820 recordable images. After two physical shutter exposures, those
  counts were 37 and 818 respectively: exactly one phone transfer per exposure.
- The app added two `Camera` items to the fixed filmstrip while live view stayed
  connected. No fatal, native, or OpenCV errors were logged.
- `SONY_20260710_221300_727.jpg` and `SONY_20260710_221313_949.jpg` are both
  6000x4000 JPEGs with `SONY / ILCE-6300` EXIF. Postview remained `Original`.
- The listener baselines stale Sony events before publishing Connected, ignores
  API-triggered capture echoes, and retries a failed physical-image download up
  to three times.
- With the listener active, one app-shutter capture added exactly one `Photo`
  item and no extra `Camera` item. Disconnecting and reconnecting without a new
  exposure kept the phone file count unchanged, confirming stale events are not
  imported.
- Final automated count after this feature: 43 JVM tests and two connected S25
  instrumentation tests, all passing.

## Continuous Shooting Verification (2026-07-10/11)

- The A6300 reports camera-service versions `1.0` through `1.4`; method discovery
  explicitly advertises `getEvent` v1.2 and continuous postview events.
- Changing Drive from Single to Continuous dynamically exposed start, stop, and
  continuous-speed APIs.
- A press-and-hold app burst produced one v1.2 Continuous event containing two
  unique 2M URLs. Both 1616x1080 JPEGs downloaded and appeared in order.
- A four-frame Burst Live ND used event batches of three and two URLs. Four were
  averaged, one extra remained saved, and the result completed in about 15 seconds
  at 1616x1080 with `SONY / ILCE-6300` EXIF.
- The current build treats each event batch as complete only after every accepted
  URL has either imported or exhausted its retries. Re-test a split A6300 burst
  and confirm diagnostics contain `continuous_capture_batch_completed` with
  `reported_url_count`, `imported_count`, and `failed_download_count`.
- Re-test two Burst Live ND runs with unchanged burst speed and shutter speed.
  The second run should use the rolling command-time estimate and should not use
  event-arrival timestamps as a capture-rate measurement.
- If `getAvailableContShootingSpeed` is currently advertised, verify the Burst
  speed selector lists the camera-provided candidates. It must disappear when
  the API is not advertised in the current mode.
- Burst Live ND restored `Original`, preserved the prior Continuous Drive mode,
  and resumed live view.
- A seven-second held app gesture stopped at the five-second watchdog while the
  synthetic finger remained down. Backgrounding another burst after one second
  ended capture after one 2M frame and the app resumed connected live view.

## Storage And Failure Modes

- Confirm new JPEGs appear in `Pictures/Alpha Capture Lab`, legacy
  `Pictures/Sony Remote` captures remain visible, and both retain EXIF metadata.
- Fill phone storage and camera card independently and confirm actionable errors.
- Interrupt a JPEG download and confirm no pending/broken gallery item remains.
- Export diagnostics and confirm there are no image bytes or Wi-Fi credentials.

## Android 16 Restricted-Mode Test

```bash
adb shell am compat enable RESTRICT_LOCAL_NETWORK com.ryu.sonyremote
adb reboot
```

With restriction enabled, verify SSDP and HTTP fail when Nearby devices is
denied and succeed after permission is granted.
