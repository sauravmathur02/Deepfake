# Comprehensive Deepfake Test Library (131 Media Files)

This folder contains **131 curated real and fake images and videos** ready for manual testing in the Deepfake Detector web dashboard.

---

## Media Breakdown

| Folder | Count | Media Type | Description / Origin |
| :--- | :---: | :---: | :--- |
| [`real_images/`](file:///d:/Projects/Deepfake/manual_test_library/real_images) | **35** | Images (`.jpg`, `.jpeg`, `.png`) | Studio portraits, official presidential/historical portraits (Obama, Biden, Kamala, Ford, Einstein), Unsplash real portraits, and real face datasets |
| [`fake_images/`](file:///d:/Projects/Deepfake/manual_test_library/fake_images) | **30** | Images (`.jpg`, `.png`) | AI diffusion generations (Midjourney, Stable Diffusion, DALL-E), HiDF deepfake face-swaps, and synthetic artifact images |
| [`real_videos/`](file:///d:/Projects/Deepfake/manual_test_library/real_videos) | **33** | Videos (`.mp4`) | Real authentic human speeches, meetings, interviews, kitchen discussions, and walking pedestrians (FaceForensics++ & UniDataPro) |
| [`fake_videos/`](file:///d:/Projects/Deepfake/manual_test_library/fake_videos) | **33** | Videos (`.mp4`) | Facial reenactment, deepfake face swaps, and AI facial motion generation (FaceForensics++ & UniDataPro) |

**Total Library Size: 65 Images + 66 Videos = 131 Media Files**

---

## How to Test

1. **Verify Server is Running**:
   - The server is running at [http://localhost:8000](http://localhost:8000). (Or start anytime via [`run.bat`](file:///d:/Projects/Deepfake/run.bat)).
2. **Open the Web Interface**:
   - Go to [http://localhost:8000](http://localhost:8000) in your browser.
3. **Drag & Drop Media**:
   - Select any image or video from the folders above and drop it into the upload box.
   - You can also upload multiple files at once for **Bulk Analysis**!
