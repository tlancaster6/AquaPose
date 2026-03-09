# Label Studio Setup for Pseudo-Label Review

## Installation

```bash
pip install label-studio
```

## Local File Serving (Windows)

Label Studio auto-enables local file serving when launched from a directory named `label-studio-data`.

1. Create `C:\Users\tucke\label-studio-data\`
2. Copy/symlink your dataset directories inside it (e.g. `round1_selected/`)
3. Launch Label Studio from that directory:
   ```powershell
   cd C:\Users\tucke\label-studio-data
   label-studio start
   ```

Image URLs use the format `/data/local-files/?d=<path-from-document-root>`, e.g.
`/data/local-files/?d=round1_selected/obb/images/train/000000_e3v82e0.jpg`

### Alternative: explicit env vars

```powershell
$env:LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED = "true"
$env:LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT = "C:\Users\tucke\label-studio-data"
label-studio start
```

## Generating Pre-Annotations

The `pseudo-label select` command generates Label Studio JSON when `--labelstudio-prefix` is provided:

```bash
aquapose pseudo-label select \
  --pseudo-dir ~/aquapose/projects/YH/runs/<run>/pseudo_labels \
  --output-dir ~/aquapose/projects/YH/training_data/round1_selected \
  --labelstudio-prefix round1_selected
```

This creates `obb/labelstudio.json` and `pose/labelstudio.json` inside the output directory.

## Creating a Project (OBB)

1. Create a new project in Label Studio
2. Set the labeling config:
   ```xml
   <View>
     <RectangleLabels name="label" toName="image" canRotate="true">
       <Label value="fish"/>
     </RectangleLabels>
     <Image name="image" value="$image"/>
   </View>
   ```
3. **Import** the `obb/labelstudio.json` file (this creates tasks with pre-annotations)
4. Go to **Settings → Cloud Storage → Add Source Storage → Local Files**
5. Set absolute local path to `C:\Users\tucke\label-studio-data\round1_selected\obb\images`
6. Save (don't sync — images are already referenced by the JSON)

## Creating a Project (Pose)

Same as OBB but with a different labeling config:

```xml
<View>
  <RectangleLabels name="label" toName="image">
    <Label value="fish"/>
  </RectangleLabels>
  <KeyPointLabels name="keypoint" toName="image">
    <Label value="nose"/>
    <Label value="head"/>
    <Label value="spine1"/>
    <Label value="spine2"/>
    <Label value="spine3"/>
    <Label value="tail"/>
  </KeyPointLabels>
  <Image name="image" value="$image"/>
</View>
```

Point cloud storage at `C:\Users\tucke\label-studio-data\round1_selected\pose\images`.

## Importing Corrections Back

After reviewing/correcting annotations in Label Studio:

1. **Export from Label Studio**: On the project page, click **Export** and choose **JSON** format. Save the file.

2. **Convert to YOLO format**:
   ```bash
   aquapose pseudo-label from-labelstudio /path/to/exported.json \
     --output-dir /tmp/corrected_obb \
     -t obb \
     --images-dir ~/aquapose/projects/YH/training_data/round1_selected/obb/images
   ```
   For pose, use `-t pose` and point `--images-dir` at the pose images directory.

3. **Import into the store**:
   ```bash
   aquapose data import --store obb --source corrected --input-dir /tmp/corrected_obb
   ```
   The `corrected` source priority is higher than `pseudo`, so corrected labels replace the originals on dedup.

4. **Re-assemble and retrain** as usual:
   ```bash
   aquapose data assemble --store obb --name round1-curated ...
   ```

## Key Gotchas

- **Import JSON, don't sync it**: Cloud storage is for serving images only. Import the JSON directly via the project Import button.
- **Don't sync cloud storage before importing**: If you sync first, Label Studio creates empty tasks from the images. Then importing the JSON creates duplicate tasks.
- **Predictions are read-only** until accepted: Click the prediction to accept it, which converts it to an editable annotation.
- **Document root matters**: The `?d=` path in image URLs is relative to the document root (`label-studio-data/`), not to the cloud storage absolute path.

## CVAT Alternative

CVAT has Ultralytics YOLO OBB/Pose import support (docs: https://docs.cvat.ai/docs/dataset_management/formats/format-yolo-ultralytics/) but as of early 2026 the online version at app.cvat.ai fails with `datumaro ImportFail`. Docker self-hosted may work with a newer version. CVAT also lacks a native rotated rectangle annotation tool — OBBs become freeform polygons.
