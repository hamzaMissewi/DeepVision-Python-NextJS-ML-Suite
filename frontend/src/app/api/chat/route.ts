// ============================================
// FILE: /app/api/chat/route.ts
// ============================================

import { OpenAIStream, StreamingTextResponse } from "ai";
import OpenAI from "openai";

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

export const runtime = "edge";

export async function POST(req: Request) {
  try {
    const { messages } = await req.json();

    const response = await openai.chat.completions.create({
      model: "gpt-3.5-turbo",
      stream: true,
      messages: [
        {
          role: "system",
          content:
            "You are a helpful AI assistant. Provide clear, concise, and accurate responses.",
        },
        ...messages,
      ],
      temperature: 0.7,
      max_tokens: 1000,
    });

    const stream = OpenAIStream(response);
    return new StreamingTextResponse(stream);
  } catch (error) {
    console.error("Chat API Error:", error);
    return new Response("Error processing request", { status: 500 });
  }
}
