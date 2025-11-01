import React from "react";
import Dashboard from "./pages/Dashboard";
import Login from "./components/Login";

function App() {
  const [user, setUser] = React.useState(null);
  return user ? <Dashboard user={user} /> : <Login onLogin={setUser} />;
}

export default App;
