import React, { useState } from "react";
import UploadBox from "../components/UploadBox";

export default function Dashboard({ user }) {
  const [count, setCount] = useState(null);

  const handleUpload = async (file) => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    setCount(data.count);
  };

  return (
    <div>
      <h3>Welcome {user.username}</h3>
      <UploadBox onUpload={handleUpload} />
      {count !== null && <p>Total Cattle Count: {count}</p>}
    </div>
  );
}
