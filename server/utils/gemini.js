const { GoogleGenAI } = require("@google/genai");
const groq = require("./groq");

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
 * Handle a chat session conversation with error fallback (Groq Llama 3.3 -> Gemini 2.5/2.0 Flash -> Local Deterministic)
 * @param {Array} messages - Chat messages in format [{ role: "user" | "assistant", content: "..." }]
 * @param {object} options - Optional config
 */
async function chat(messages, options = {}) {
    // 1. Try Groq (Llama 3.3 70B) if key is present
    const groqKey = process.env.GROQ_API_KEY;
    if (groqKey) {
        try {
            const reply = await groq.chat(messages, options);
            if (reply) return reply;
        } catch (error) {
            console.error("[Fallback Pipeline] Groq Chat failed, falling back to Gemini:", error.message);
        }
    }

    // 2. Try Gemini (2.5/2.0 Flash)
    const geminiKey = process.env.GEMINI_API_KEY;
    const isGeminiExhausted = !geminiKey || (geminiKey.startsWith("AQ.Ab8RN6") && geminiKey.length === 52);

    if (!isGeminiExhausted) {
        try {
            const client = getAIClient();
            const model = options.model || process.env.GEMINI_MODEL || "gemini-2.5-flash";
            
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

            if (response.text) return response.text;
        } catch (error) {
            console.error("[Fallback Pipeline] Gemini Chat failed, falling back to local deterministic responses:", error.message);
        }
    }

    // 3. Local Deterministic Fallback (reliability without API dependency)
    return getMockResponse(messages, isGeminiExhausted || !!groqKey);
}

// Generate an intelligent mock response for testing/fallback
function getMockResponse(messages, isQuotaExceeded = false) {
    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user')?.content || '';
    const query = lastUserMessage.toLowerCase();
    
    let prefix = "";
    if (process.env.NODE_ENV === 'production') {
        prefix = `⚠️ **Offline Demo Mode** (API keys are not configured or failed)\n\n`;
    } else {
        if (!isQuotaExceeded) {
            prefix = `⚠️ **Demo Mode** (No active Gemini/Groq API Key found in \`server/.env\`)\n\n`;
        } else {
            prefix = `⚠️ **Fallback Mode** (API quota exceeded or key invalid)\n\n`;
        }
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
    
    if (process.env.NODE_ENV === 'production') {
        return prefix + `I received your prompt: "${lastUserMessage}"\n\nTo get live answers from the AI, please configure your \`GROQ_API_KEY\` or \`GEMINI_API_KEY\` in your environment variables.`;
    }
    return prefix + `I received your prompt: "${lastUserMessage}"\n\nTo get live answers from Groq (Llama 3.3) or Gemini (2.5 Pro / Flash), please edit \`server/.env\` and add your API keys.`;
}

module.exports = {
    getAIClient,
    generateText,
    chat
};
