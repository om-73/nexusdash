const axios = require('axios');

/**
 * Generate chat response using Groq Llama 3.3 70B
 * @param {Array} messages - Chat messages in format [{ role: "user" | "assistant", content: "..." }]
 * @param {object} options - Optional parameters (e.g. systemInstruction, model)
 */
async function chat(messages, options = {}) {
    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
        throw new Error("GROQ_API_KEY is not configured in the environment.");
    }

    const model = options.model || "llama-3.3-70b-versatile";
    const endpoint = "https://api.groq.com/openai/v1/chat/completions";

    // Format messages for Groq API (standard OpenAI format)
    const formattedMessages = [];
    
    // Add system instruction if provided
    if (options.systemInstruction) {
        formattedMessages.push({
            role: "system",
            content: options.systemInstruction
        });
    }

    // Map and append the conversational history
    messages.forEach(msg => {
        formattedMessages.push({
            role: msg.role === "model" ? "assistant" : msg.role,
            content: msg.content
        });
    });

    try {
        console.log(`[Groq] Sending request to model ${model} with ${formattedMessages.length} messages...`);
        const response = await axios.post(
            endpoint,
            {
                model: model,
                messages: formattedMessages,
                temperature: options.temperature !== undefined ? options.temperature : 0.7,
                max_completion_tokens: options.maxTokens || 1024
            },
            {
                headers: {
                    "Content-Type": "application/json",
                    "Authorization": `Bearer ${apiKey}`
                },
                timeout: 10000 // 10s timeout
            }
        );

        if (response.data && response.data.choices && response.data.choices[0]) {
            const reply = response.data.choices[0].message.content;
            return reply;
        } else {
            throw new Error("Invalid response format received from Groq API.");
        }
    } catch (error) {
        console.error("Groq API Error:", error.response?.data || error.message);
        throw error;
    }
}

module.exports = {
    chat
};
