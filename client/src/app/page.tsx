'use client';

import { Fragment, useState, useEffect } from 'react';
import { useChat } from '@ai-sdk/react';
import { ArrowUp, CopyIcon, Loader2Icon, ChevronLeft, MessageSquare, Fish } from 'lucide-react';

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
import MapboxMap from '@/components/Map';
import SharkSidebar from '@/components/SharkSidebar';
import TimeSlider from '@/components/TimeSlider';

/**
 * Main interface with shark zone visualization:
 * - Left: Shark species selection sidebar
 * - Center: Interactive Mapbox map showing shark zones
 * - Right: AI chatbot for additional information
 */
export default function SharkZoneViewer() {
  const [input, setInput] = useState('');
  const { messages, sendMessage, status } = useChat();
  
  // Sidebar state management
  const [leftSidebarOpen, setLeftSidebarOpen] = useState(true);
  const [rightSidebarOpen, setRightSidebarOpen] = useState(false);
  const [temperature, setTemperature] = useState(0.7);
  
  // Shark selection state
  const [selectedShark, setSelectedShark] = useState<any>(null);
  
  // Time state
  const [selectedYear, setSelectedYear] = useState(2024);
  const [selectedMonth, setSelectedMonth] = useState(6);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playInterval, setPlayInterval] = useState<NodeJS.Timeout | null>(null);

  const handleSubmit = (message: { text: string }) => {
    if (!message.text) return;

    sendMessage({
      text: message.text,
    });
    setInput('');
  };

  // Time control functions
  const handlePlayPause = () => {
    if (isPlaying) {
      if (playInterval) {
        clearInterval(playInterval);
        setPlayInterval(null);
      }
      setIsPlaying(false);
    } else {
      const interval = setInterval(() => {
        setSelectedMonth(prev => {
          if (prev >= 12) {
            setSelectedYear(prevYear => prevYear + 1);
            return 1;
          }
          return prev + 1;
        });
      }, 2000); // Change month every 2 seconds
      setPlayInterval(interval);
      setIsPlaying(true);
    }
  };

  const handleReset = () => {
    if (playInterval) {
      clearInterval(playInterval);
      setPlayInterval(null);
    }
    setIsPlaying(false);
    setSelectedYear(2024);
    setSelectedMonth(6);
  };

  // Cleanup interval on unmount
  useEffect(() => {
    return () => {
      if (playInterval) {
        clearInterval(playInterval);
      }
    };
  }, [playInterval]);

  return (
    <div className="w-full h-screen relative bg-background">
      {/* Shark Sidebar */}
      <SharkSidebar
        selectedShark={selectedShark}
        onSharkSelect={setSelectedShark}
        isOpen={leftSidebarOpen}
        onToggle={() => setLeftSidebarOpen(!leftSidebarOpen)}
      />
      
      {/* Map Section - Always full width */}
      <div className="w-full h-full relative">
        <MapboxMap 
          className="h-full w-full" 
          selectedShark={selectedShark}
          selectedYear={selectedYear}
          selectedMonth={selectedMonth}
        />
        
        {/* Toggle Buttons - Always visible on map */}
        <button
          onClick={() => setLeftSidebarOpen(!leftSidebarOpen)}
          className="absolute left-4 top-1/2 -translate-y-1/2 z-30 bg-card border border-border rounded-md p-2 hover:bg-accent transition-colors shadow-lg"
          aria-label="Toggle shark sidebar"
        >
          {leftSidebarOpen ? <ChevronLeft className="size-4" /> : <Fish className="size-4" />}
        </button>
        
        <button
          onClick={() => setRightSidebarOpen(!rightSidebarOpen)}
          className="absolute right-4 top-1/2 -translate-y-1/2 z-30 bg-card border border-border rounded-md p-2 hover:bg-accent transition-colors shadow-lg"
          aria-label="Toggle chat sidebar"
        >
          {rightSidebarOpen ? <ChevronLeft className="size-4" /> : <MessageSquare className="size-4" />}
        </button>
      </div>

      {/* Right Sidebar - Chat Interface - Absolute positioned overlay */}
      <div className={`absolute top-0 right-0 h-full bg-card border-l border-border z-20 transition-transform duration-300 ease-in-out ${rightSidebarOpen ? 'translate-x-0' : 'translate-x-full'} w-96`}>
        <div className="flex flex-col h-full">
          {/* Header */}
          <div className="flex items-center justify-between p-4 border-b border-border bg-card/50">
            <div className="flex items-center gap-2">
              <MessageSquare className="size-5" />
              <h2 className="text-lg font-semibold">AI Assistant</h2>
            </div>
          </div>
          
          {/* Messages Area */}
          <div className="flex-1 min-h-0 p-4">
            <Conversation className="h-full">
              <ConversationContent className="space-y-4">
                {messages.length === 0 ? (
                  <div className="flex items-center justify-center h-full text-center">
                    <div className="space-y-2">
                      <MessageSquare className="size-12 mx-auto text-muted-foreground" />
                      <p className="text-muted-foreground">Ask about shark zones</p>
                      <p className="text-sm text-muted-foreground">
                        Get information about shark species, SST data, or zone details
                      </p>
                    </div>
                  </div>
                ) : (
                  messages.map((message) => (
                    <div key={message.id} className="space-y-2">
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
                  ))
                )}
                {status === 'submitted' && <Loader />}
              </ConversationContent>
              <ConversationScrollButton />
            </Conversation>
          </div>

          {/* Input area - Fixed at bottom */}
          <div className="p-4 border-t border-border bg-card/50">
            <form 
              onSubmit={(e) => {
                e.preventDefault();
                if (input.trim()) {
                  handleSubmit({ text: input });
                }
              }}
              className="space-y-2"
            >
              <div className="relative">
                <textarea
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about shark zones, SST data, or species information..."
                  className="w-full resize-none border border-border rounded-lg p-3 pr-12 bg-background text-sm placeholder:text-muted-foreground min-h-[2.5rem] max-h-32 focus:outline-none focus:ring-2 focus:ring-primary focus:border-transparent"
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
                  className="absolute right-2 bottom-2 flex items-center justify-center w-8 h-8 rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
                >
                  {status === 'submitted' ? (
                    <Loader2Icon className="size-4 animate-spin" />
                  ) : (
                    <ArrowUp className="size-4" />
                  )}
                </button>
              </div>
            </form>
          </div>
        </div>
      </div>
      
      {/* Time Slider */}
      <TimeSlider
        selectedYear={selectedYear}
        selectedMonth={selectedMonth}
        onYearChange={setSelectedYear}
        onMonthChange={setSelectedMonth}
        isPlaying={isPlaying}
        onPlayPause={handlePlayPause}
        onReset={handleReset}
      />
    </div>
  );
}


