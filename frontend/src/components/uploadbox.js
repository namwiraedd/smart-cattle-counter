import React from "react";

export default function UploadBox({ onUpload }) {
  return (
    <div
      onDrop={(e) => {
        e.preventDefault();
        onUpload(e.dataTransfer.files[0]);
      }}
      onDragOver={(e) => e.preventDefault()}
      className="upload-box"
    >
      Drag & drop image or video here
    </div>
  );
}
