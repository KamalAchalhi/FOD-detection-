# Foreign Object Debris (FOD) Detection

## 1. Methodology

This section details the different steps implemented to achieve the set objective, applied to a car engine. The main idea is to create a 3D model serving as an ideal reference for the engine, without added or missing parts. This model serves as a comparison base to detect any foreign or missing objects.

To do this, we compare images of the reference model with those of the modified engine. The detection of added or missing objects, considered as FODs (Foreign Object Debris), thus requires images of the engine after modification. These images will help identify differences and precisely segment added or missing objects.

## 2. Data Acquisition, Preprocessing, and 3D Reconstruction with Metashape

### 2.1 Photo Capture
We took two types of images:
- Photos for constructing a 3D model.
- Photos containing FODs.

We have 637 images without FODs of a car engine. These images have a resolution of 6048 × 4024 and are in JPG format.

### 2.2 3D Reconstruction with Metashape
The 3D reconstruction in Metashape is based on photogrammetric methods.

We used the FOD-free photos to build a 3D model in Metashape. This step provides:
- Coordinates (translation vector, rotation matrix).
- The intrinsic parameters of the camera used.

Data export:
```sh
Export > Cameras > XML Format
```

### 2.3 Alignment of Images Containing FODs and Synthesis of New Views
We first imported 45 images containing FODs into the chunk of our model. Then, we aligned these images with the model.

We synthesized new views using the camera positions of the FOD-containing images. A Python script executed in the Metashape console allows:
- Manipulating the model.
- Using methods such as chunk.cameras.render.
- Selecting cameras from FOD-containing images.

This synthesis through photogrammetry will be compared with NeRFs and Gaussian Splatting methods.

## 3. Camera Data Export and Comparison with NerfStudio

We exported camera data for images containing FODs. These data are essential for training NeRFs (Neural Radiance Fields) in NerfStudio.

Why use Metashape instead of COLMAP?
- Metashape is more precise than COLMAP, making it a better choice for our project.

## 4. Conventions and Reference Frames

### 4.1 Metashape Convention Description
- X-axis: horizontal
- Y-axis: depth
- Z-axis: vertical (up/down)

### 4.2 Transformation Matrix and Camera Reference Frame
Each camera has a reference frame defined by a transformation matrix:
- X: horizontal direction to the right
- Y: vertical direction downward
- Z: optical axis direction (in front of the camera)

## 5. Exported Data Formalism

### 5.1 XML File Structure
The exported XML file contains several sections:
- Chunk: 3D model
- Sensors: camera properties
- Cameras: 4×4 transformation matrix (position and orientation)

### 5.2 Exported Data Analysis
We analyzed this file to understand the differences in reference frames between Metashape and NeRFs.

Essential data for training a NeRF:
- Transformation matrix (position and orientation)
- Camera orientation
- Camera intrinsic parameters

## 6. From Metashape Reference Frame to NerfStudio Reference Frame

NerfStudio structures the data for training NeRFs.

Command used:
```sh
ns-process-data metashape
```

This allows:
- Converting the XML file to JSON.
- Generating the necessary directories for training.

## 7. Image Synthesis with FODs

We synthesized views by retrieving the model trained with FOD-free images.

Command to generate synthetic images:
```sh
ns-render camera path --camera_path camera_path.json
```

This generates new views based on the aligned camera positions.

## 8. Matrix Transformation

Each transformation follows the formula:

\[
T' = \begin{bmatrix} r_{11} & r_{12} & r_{13} & t_1 \\ r_{21} & r_{22} & r_{23} & t_2 \\ r_{31} & r_{32} & r_{33} & t_3 \\ 0 & 0 & 0 & 1 \end{bmatrix}
\]

where:
- \( T' \) is the **final matrix** after transformation.
- \( R' \) is the **new rotation matrix**, obtained after regularization and axis inversion.
- \( t' \) is the **translation vector**, scaled using the **scale factor**.

The matrix product is then computed as:

\[
M \times T
\]

where:
- \( M \) is the **regularization matrix** from `dataparser_transforms.json`.
- \( T \) is the **transformation matrix** extracted from Metashape's XML.
- \( e \) is the **scale factor** applied to translation:

\[
t' = t \times e
\]


