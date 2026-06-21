const { GoogleGenAI } = require("@google/genai");

let ai = null;

function getAIClient() {
    if (!ai) {
        const apiKey = process.env.GEMINI_API_KEY;
        if (!apiKey) {
            console.warn("WARNING: GEMINI_API_KEY is not set in environment variables.");
        }
        ai = new GoogleGenAI({ apiKey });
    }
    return ai;
}

/**
 * Generate content using Gemini 2.5 Pro / Flash
 * @param {string} prompt - User prompt
 * @param {object} options - Optional parameters (e.g. systemInstruction, model)
 */
async function generateText(prompt, options = {}) {
    const client = getAIClient();
    try {
        const model = options.model || process.env.GEMINI_MODEL || "gemini-2.5-flash";
        const config = {};
        
        if (options.systemInstruction) {
            config.systemInstruction = options.systemInstruction;
        }
        
        const response = await client.models.generateContent({
            model: model,
            contents: prompt,
            config: config
        });
        
        return response.text;
    } catch (error) {
        console.error("Gemini Utility Error:", error);
        throw error;
    }
}

/**
 * Handle a chat session conversation with error fallback
 * @param {Array} messages - Chat messages in format [{ role: "user" | "assistant", content: "..." }]
 * @param {object} options - Optional config
 */
async function chat(messages, options = {}) {
    const apiKey = process.env.GEMINI_API_KEY;
    
    // Check if key is the exhausted default key or missing
    const isExhaustedKey = !apiKey || (apiKey.startsWith("AQ.Ab8RN6") && apiKey.length === 52);
    
    if (isExhaustedKey) {
        if (process.env.NODE_ENV === 'production') {
            return "I'm sorry, I encountered an issue connecting to my brain. Please verify that your Gemini API key is configured correctly in the environment.";
        }
        return getMockResponse(messages, false);
    }

    try {
        const client = getAIClient();
        const model = options.model || process.env.GEMINI_MODEL || "gemini-2.5-flash";
        
        // Map messages format to Gemini format
        const contents = messages.map(msg => ({
            role: msg.role === "assistant" ? "model" : "user",
            parts: [{ text: msg.content }]
        }));
        
        const config = {};
        if (options.systemInstruction) {
            config.systemInstruction = options.systemInstruction;
        }

        const response = await client.models.generateContent({
            model: model,
            contents: contents,
            config: config
        });

        return response.text;
    } catch (error) {
        console.error("Gemini Chat Utility Error:", error);
        
        if (process.env.NODE_ENV === 'production') {
            return "I'm sorry, I encountered an issue connecting to my brain. Please verify that your Gemini API key is configured correctly in the environment.";
        }
        
        // Handle Quota/API errors gracefully with a fallback message rather than throwing 500
        if (error.status === 429 || (error.message && error.message.includes("quota"))) {
            return `⚠️ **Gemini API Quota Exceeded (429)**\n\nThe API key configured in \`server/.env\` has a quota limit of 0 or has run out of requests for today. Please update \`server/.env\` with a new key from [Google AI Studio](https://aistudio.google.com/).\n\n**Simulated Assistant Fallback:**\n${getMockResponse(messages, true)}`;
        }
        
        throw error;
    }
}

// Generate an intelligent mock response for testing/fallback
function getMockResponse(messages, isQuotaExceeded = false) {
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    const query = lastUserMessage.toLowerCase();
    
    let prefix = "";
    if (!isQuotaExceeded) {
        prefix = `⚠️ **Demo Mode** (No active Gemini API Key found in \`server/.env\`)\n\n`;
    }
    
    if (query.includes("clean") || query.includes("missing") || query.includes("null") || query.includes("drop") || query.includes("impute")) {
        return prefix + "To clean your dataset and handle missing values, navigate to the **Data Cleaning** tab on the left sidebar. You can choose to drop columns, drop rows, or fill missing fields with the Median/Mean/Mode strategy in one click.";
    }
    if (query.includes("chart") || query.includes("visualize") || query.includes("plot") || query.includes("graph") || query.includes("distribution") || query.includes("correlation")) {
        return prefix + "For data visualization, use the **Exploratory Analysis** tab. Based on typical data types:\n- **Numerical columns**: Best suited for line or bar charts to show distributions.\n- **Categorical columns**: Best suited for bar charts or pie charts.";
    }
    if (query.includes("model") || query.includes("machine learning") || query.includes("predict") || query.includes("pridict") || query.includes("train") || query.includes("fit") || query.includes("regression") || query.includes("classification")) {
        return prefix + "NexusDash supports training Regression and Classification models. Navigate to the **Machine Learning** tab, specify your target column, and click train. Once completed, you can test it live in the **Use Model (Predict)** tab.";
    }
    if (query.includes("dataset") || query.includes("structure") || query.includes("columns")) {
        return prefix + "I can see the active dataset metadata. You have successfully loaded your dataset into active memory! You can use the **Overview** dashboard to view the column types, shapes, and memory usage.";
    }
    
    return prefix + `I received your prompt: "${lastUserMessage}"\n\nTo get live answers from the Gemini 2.5 Pro / Flash model, please edit \`server/.env\` and replace \`GEMINI_API_KEY\` with a valid, active API key.`;
}

module.exports = {
    getAIClient,
    generateText,
    chat
};
