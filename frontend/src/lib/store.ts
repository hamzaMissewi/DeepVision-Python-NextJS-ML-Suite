/**
 * FULL-STACK AI-POWERED CHAT APPLICATION
 * Trending Topic: AI Chatbot with Real-time Streaming
 *
 * SETUP INSTRUCTIONS:
 *
 * 1. Create Next.js app:
 *    npx create-next-app@latest ai-chat-app --typescript --tailwind --app
 *
 * 2. Install dependencies:
 *    npm install ai openai zod zustand lucide-react
 *
 * 3. Setup environment (.env.local):
 *    OPENAI_API_KEY=your_openai_key_here
 *
 * 4. File structure:
 *    /app
 *      /api
 *        /chat
 *          route.ts (this file - API route)
 *      page.tsx (this file - main page)
 *      layout.tsx
 *    /lib
 *      store.ts (this file - state management)
 *
 * Features:
 * - Real-time AI streaming responses
 * - Conversation history
 * - Modern UI with Tailwind
 * - State management with Zustand
 * - OpenAI GPT integration
 */

// ============================================
// FILE: /lib/store.ts
// ============================================

import { create } from "zustand";
import { persist } from "zustand/middleware";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

interface ChatStore {
  messages: Message[];
  isLoading: boolean;
  addMessage: (message: Omit<Message, "id" | "timestamp">) => void;
  clearMessages: () => void;
  setLoading: (loading: boolean) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set) => ({
      messages: [],
      isLoading: false,

      addMessage: (message) =>
        set((state) => ({
          messages: [
            ...state.messages,
            {
              ...message,
              id: crypto.randomUUID(),
              timestamp: new Date(),
            },
          ],
        })),

      clearMessages: () => set({ messages: [] }),

      setLoading: (loading) => set({ isLoading: loading }),
    }),
    {
      name: "chat-storage",
    }
  )
);
