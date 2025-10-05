'use client';

import { Fragment, useState, useEffect } from 'react';
import { useChat } from '@ai-sdk/react';
import { ArrowUp, CopyIcon, Loader2Icon } from 'lucide-react';

// Type definitions for tool results
interface ToolResultWithRefresh {
  refreshMap?: boolean;
}

interface ToolResultWithMapAction {
  action?: string;
  location?: {
    name: string;
    latitude: number;
    longitude: number;
    zoom: number;
  };
}

// AI Components
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from '@/components/ai-elements/conversation';
import { Message, MessageContent } from '@/components/ai-elements/message';
import { Response } from '@/components/ai-elements/response';
import { Action, Actions } from '@/components/ai-elements/actions';
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from '@/components/ai-elements/reasoning';
import {
  Sources,
  Source,
  SourcesContent,
  SourcesTrigger,
} from '@/components/ai-elements/sources';
import { Loader } from '@/components/ai-elements/loader';

// Custom Components
import RotatingEarth from '@/components/Globe';

/**
 * Main chat interface with split view:
 * - Left: Interactive 3D Globe
 * - Right: AI chatbot using GPT-5
 */
export default function GlobeChatBotDemo() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, status } = useChat();

  const handleSubmit = (message: { text: string }) => {
    if (!message.text) return;

    sendMessage({
      text: message.text,
    });
    setInput('');
  };

  return (
    <div className="w-full h-screen relative">
      {/* Split layout: Globe on left, Chat on right */}
      <div className="flex h-full">
        
        {/* Globe Section - Takes up more space */}
        <div className="flex-[4] h-full">
          <RotatingEarth className="h-full w-full" />
        </div>

        {/* Chat Section - Takes up less space */}
        <div className="flex-[1] min-h-0 flex flex-col max-w-sm p-4">
          <div className="flex-1 min-h-0">
            <Conversation className="h-full">
              <ConversationContent>
                {messages.map((message) => (
                  <div key={message.id}>
                    {/* Sources (for web search results) */}
                    {message.role === 'assistant' && message.parts.filter((part) => part.type === 'source-url').length > 0 && (
                      <Sources>
                        <SourcesTrigger
                          count={
                            message.parts.filter(
                              (part) => part.type === 'source-url',
                            ).length
                          }
                        />
                        {message.parts.filter((part) => part.type === 'source-url').map((part, i) => (
                          <SourcesContent key={`${message.id}-${i}`}>
                            <Source
                              key={`${message.id}-${i}`}
                              href={part.url}
                              title={part.url}
                            />
                          </SourcesContent>
                        ))}
                      </Sources>
                    )}

                    {/* Message parts (text, reasoning, etc.) */}
                    {message.parts.map((part, i) => {
                      switch (part.type) {
                        case 'text':
                          return (
                            <Fragment key={`${message.id}-${i}`}>
                              <Message from={message.role}>
                                <MessageContent>
                                  <Response>
                                    {part.text}
                                  </Response>
                                </MessageContent>
                              </Message>
                              {/* Copy button for last assistant message */}
                              {message.role === 'assistant' && i === message.parts.length - 1 && (
                                <Actions className="mt-2">
                                  <Action
                                    onClick={() => navigator.clipboard.writeText(part.text)}
                                    label="Copy"
                                  >
                                    <CopyIcon className="size-3" />
                                  </Action>
                                </Actions>
                              )}
                            </Fragment>
                          );
                        case 'reasoning':
                          return (
                            <Reasoning
                              key={`${message.id}-${i}`}
                              className="w-full"
                              isStreaming={status === 'streaming' && i === message.parts.length - 1 && message.id === messages.at(-1)?.id}
                            >
                              <ReasoningTrigger />
                              <ReasoningContent>{part.text}</ReasoningContent>
                            </Reasoning>
                          );
                        default:
                          return null;
                      }
                    })}
                  </div>
                ))}
                {status === 'submitted' && <Loader />}
              </ConversationContent>
              <ConversationScrollButton />
            </Conversation>
          </div>

          {/* Input area */}
          <div className="mt-4">
            <form 
              onSubmit={(e) => {
                e.preventDefault();
                if (input.trim()) {
                  handleSubmit({ text: input });
                }
              }}
              className="relative flex items-end gap-2 p-3 border border-border rounded-xl bg-background shadow-sm"
            >
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="What would you like to know about the world?"
                className="flex-1 resize-none border-none outline-none bg-transparent text-sm placeholder:text-muted-foreground max-h-32 min-h-[2.5rem] field-sizing-content"
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    if (input.trim()) {
                      handleSubmit({ text: input });
                    }
                  }
                }}
              />
              <button
                type="submit"
                disabled={!input.trim() || status === 'submitted'}
                className="flex items-center justify-center w-8 h-8 rounded-lg bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {status === 'submitted' ? (
                  <Loader2Icon className="size-4 animate-spin" />
                ) : (
                  <ArrowUp className="size-4" />
                )}
              </button>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
