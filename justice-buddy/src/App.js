import React, { useState } from "react";
import "./App.css";

function App() {
  const [chat, setChat] = useState([]);
  const [message, setMessage] = useState("");

  const sendMessage = async () => {
    if (!message.trim()) return;
  
    const userMessage = { sender: "user", text: message };
    setChat((prevChat) => [...prevChat, userMessage]);
    setMessage("");

    try {
      const response = await fetch("http://127.0.0.1:5000/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message }),
      });
  
      const data = await response.json();
      addBotMessage(data.response);
    } catch (error) {
      console.error("Error:", error);
    }
  };

  const addBotMessage = (text) => {
    const botMessage = { sender: "bot", text: "", fullText: text };
    setChat((prevChat) => [...prevChat, botMessage]);

    let index = 0;
    const interval = setInterval(() => {
      setChat((prevChat) => {
        return prevChat.map((msg, i) =>
          i === prevChat.length - 1
            ? { ...msg, text: msg.fullText.slice(0, index + 1) }
            : msg
        );
      });

      index++;
      if (index >= text.length) clearInterval(interval);
    }, 15);
  };

  return (
    <div className="chat-container">
      <div className="header">⚖ Justice Buddy</div>
      <div className="chat-box">
        {chat.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}`}>
            <span className="text">{msg.text}</span>
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          type="text"
          placeholder="Type your message..."
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              sendMessage();
            }
          }}
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}

export default App;