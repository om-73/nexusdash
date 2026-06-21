import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

// The SDK automatically looks for the GEMINI_API_KEY environment variable
const ai = new GoogleGenAI({});

async function generateText() {
  try {
    console.log("Sending query to gemini-2.5-flash...");
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash", 
      contents: "Explain the benefit of asynchronous programming in Node.js in one sentence.",
    });

    console.log("Gemini Response:", response.text);
  } catch (error) {
    console.error("Error communicating with Gemini:", error);
  }
}

generateText();
