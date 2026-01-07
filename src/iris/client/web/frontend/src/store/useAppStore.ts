import { create } from "zustand";
import { createCameraSlice, type CameraSlice } from "./slices/createCameraSlice";
import { createAnalysisSlice, type AnalysisSlice } from "./slices/createAnalysisSlice";
import { createUISlice, type UISlice } from "./slices/createUISlice";

// Combined store type
export type AppState = CameraSlice & AnalysisSlice & UISlice;

// Create the combined store using Zustand slices pattern
export const useAppStore = create<AppState>()((...a) => ({
  ...createCameraSlice(...a),
  ...createAnalysisSlice(...a),
  ...createUISlice(...a),
}));
