# Data files

`sample_revit_data.json` is a minimal schema example for the Flask--Dynamo handoff file.
It is not the full Revit model state and does not correspond to real Autodesk Revit element IDs.

For actual runs, populate `revit_data.json` from Revit/Dynamo and point `VIBE_JSON_PATH` to that file in `.env`.
