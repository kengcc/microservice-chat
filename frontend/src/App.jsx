// Copyright (c) 2024 Telekom Research and Development Sdn BHd
// 
//  Permission is hereby granted, free of charge, to any person obtaining a copy
//  of this software and associated documentation files (the "Software"), to deal
//  in the Software without restriction, including without limitation the rights
//  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//  copies of the Software, and to permit persons to whom the Software is
//  furnished to do so, subject to the following conditions:
// 
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
// 
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
//  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
//  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
//  OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
//  OR OTHER DEALINGS IN THE SOFTWARE.
// 

import React, { useState, useEffect } from "react";
import { FaSpinner } from "react-icons/fa";

const App = () => {
  const [conversation, setConversation] = useState({ conversation: [] });
  const [userMessage, setUserMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    const fetchConversation = async () => {
      const conversationId = localStorage.getItem("conversationId");
      if (conversationId) {
        const response = await fetch(
          `http://localhost:5000/chat_service/${conversationId}`
        );
        const data = await response.json();
        if (!data.error) {
          setConversation(data);
        }
      }
    };

    fetchConversation();
  }, []);

  const generateConversationId = () =>
    "_" + Math.random().toString(36).slice(2, 11);

  const handleInputChange = (event) => {
    setUserMessage(event.target.value);
  };

  const handleNewSession = () => {
    localStorage.removeItem("conversationId");
    setConversation({ conversation: [] });
  };

  const handleSubmit = async () => {
    setIsLoading(true);
    let conversationId = localStorage.getItem("conversationId");
    if (!conversationId) {
      conversationId = generateConversationId();
      localStorage.setItem("conversationId", conversationId);
    }

    const newConversation = [
      ...conversation.conversation,
      { role: "User", content: userMessage },
    ];

    const response = await fetch(
      `http://localhost:5000/chat_service/${conversationId}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ conversation: newConversation }),
      }
    );

    const data = await response.json();
    setConversation(data);
    setUserMessage("");
    setIsLoading(false);
  };

  return (
    <div
      className="App flex flex-col items-center pt-6 min-h-screen bg-gray-900 text-sm"
      style={{
        backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('/images/background.jpg')`,
        backgroundSize: "cover",
        backgroundPosition: "center",
        backgroundRepeat: "no-repeat",
      }}
    >
      <h1 className="text-3xl font-bold mb-4 text-white">BVM Real-Time Assistant</h1>
      {conversation.conversation.length > 0 && (
        <div className="flex flex-col p-4 bg-white rounded shadow w-full max-w-md space-y-4">
          {conversation.conversation
            .filter((message) => message.role !== "system")
            .map((message, index) => (
              <div
                key={index}
                className={`${
                  message.role === "User" ? "text-right" : "text-left"
                }`}
              >
                <strong className="font-bold text-gray-900">
                  {message.role}:
                </strong>
                <span className="text-gray-700">{message.content}</span>
              </div>
            ))}
        </div>
      )}
      <div className="flex flex-row w-full max-w-md mt-4">
        <input
          type="text"
          value={userMessage}
          onChange={handleInputChange}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              handleSubmit();
            }
          }}
          className="flex-grow mr-2 p-2 rounded border-gray-300"
          placeholder={isLoading ? "Processing..." : "Type your message here"}
        />
        <button
          onClick={handleSubmit}
          disabled={isLoading}
          className="p-2 rounded bg-blue-800 text-white"
        >
          Send
        </button>
      </div>
      {isLoading && (
        <div className="flex items-center space-x-4 text-l text-white">
          <FaSpinner className="animate-spin" />
          <span>Loading...</span>
        </div>
      )}
      <button
        onClick={handleNewSession}
        className="mt-4 p-2 rounded bg-red-700 text-white absolute top-0 left-4"
      >
        New Session
      </button>
    </div>
  );
};

export default App;
