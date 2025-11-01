import React from "react";

export default function Login({ onLogin }) {
  const handleLogin = () => onLogin({ username: "demo" });
  return (
    <div className="login">
      <h2>Smart Cattle Counter</h2>
      <button onClick={handleLogin}>Login</button>
    </div>
  );
}
