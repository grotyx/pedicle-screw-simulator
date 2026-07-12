# Local DICOM Data

This directory is reserved for local test data. DICOM, NIfTI, NRRD, MHA, and derived segmentation files must not be committed to the repository.

## Privacy warning

Clinical images may contain identifying information in standard attributes, private attributes, UIDs, burned-in pixel annotations, overlays, structured content, and file metadata. Removing only `PatientName` and `PatientID` is not sufficient.

Before using any dataset for screenshots, tutorials, bug reports, or public samples:

1. Obtain permission appropriate to the dataset and jurisdiction.
2. Create a separate de-identified copy; never modify the clinical original.
3. Apply a validated DICOM de-identification workflow based on the DICOM Basic Application Level Confidentiality Profile and the options required for the use case.
4. Inspect private attributes, pixel data, overlays, filenames, folder names, and exported screenshots.
5. Confirm that `PatientIdentityRemoved` and the de-identification method are recorded when appropriate.
6. Perform an independent manual review before publication.

Recommended public alternatives are synthetic phantoms or established public datasets whose terms explicitly permit redistribution.

## Local layout

```text
data/
├── README.md
├── study_001/       # local only
│   └── *.dcm
└── segmentation/    # local only
    └── *.nii.gz
```

The application does not require a fixed folder name. Use **Open DICOM Folder** and select the directory containing a DICOM series.
