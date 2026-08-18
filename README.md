# Taller NeuroCo: EEG, DICOM y NIfTI

Material introductorio para aprender a inspeccionar, visualizar y convertir datos neurocientíficos.

## Inicio rápido

Se recomienda Python 3.11, 3.12 o 3.13. Desde esta carpeta:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab NeuroCo_EEG_DICOM_NIfTI.ipynb
```

En macOS o Linux, la activación es `source .venv/bin/activate`.

El notebook incluye EEG e imagen sintéticos, el corte CT de ejemplo distribuido con pydicom y un EDF público de EEGBCI descargado de PhysioNet. La procedencia, licencia y cita del EDF están en `datos/ejemplo_eegbci/README.md`. Para trabajar con datos propios, use copias desidentificadas dentro de `datos/` siguiendo las rutas indicadas en las celdas. La conversión recomendada de DICOM a NIfTI requiere además que `dcm2niix` esté instalado y disponible en el `PATH`.

La última sección muestra cómo pasar de datos crudos a arrays listos para modelos: filtrado, segmentación, rechazo simple de artefactos, extracción de potencia por bandas, partición sin fuga, `Pipeline` de scikit-learn y preparación de volúmenes 3D para PyTorch o Keras.

## Archivos

- `NeuroCo_EEG_DICOM_NIfTI.ipynb`: notebook principal.
- `requirements.txt`: dependencias de Python.
- `build_notebook.py`: fuente reproducible que regenera el notebook.

No suba DICOM identificables ni datos sensibles al repositorio.
