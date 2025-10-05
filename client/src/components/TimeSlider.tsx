'use client';

import { useState } from 'react';
import { Calendar, Clock, Play, Pause, RotateCcw } from 'lucide-react';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';

interface TimeSliderProps {
  selectedYear: number;
  selectedMonth: number;
  onYearChange: (year: number) => void;
  onMonthChange: (month: number) => void;
  isPlaying: boolean;
  onPlayPause: () => void;
  onReset: () => void;
}

const MONTHS = [
  'January', 'February', 'March', 'April', 'May', 'June',
  'July', 'August', 'September', 'October', 'November', 'December'
];

const MIN_YEAR = 2024;
const MAX_YEAR = 2025;

export default function TimeSlider({ 
  selectedYear, 
  selectedMonth, 
  onYearChange, 
  onMonthChange, 
  isPlaying, 
  onPlayPause, 
  onReset 
}: TimeSliderProps) {
  const [isHovered, setIsHovered] = useState(false);

  return (
    <div 
      className={`absolute bottom-4 left-1/2 -translate-x-1/2 z-30 bg-card/95 backdrop-blur-sm border border-border rounded-lg p-4 min-w-[500px] max-w-[700px] time-slider-container ${
        isHovered ? 'shadow-2xl scale-105' : 'shadow-lg'
      }`}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 text-sm font-medium">
            <Calendar className="size-4" />
            <span>🔮 Shark Zone Predictions</span>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onPlayPause}
              className="flex items-center gap-1"
            >
              {isPlaying ? <Pause className="size-3" /> : <Play className="size-3" />}
              {isPlaying ? 'Pause' : 'Play'}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={onReset}
              className="flex items-center gap-1"
            >
              <RotateCcw className="size-3" />
              Reset
            </Button>
          </div>
        </div>
        
        {/* Year Slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Year: {selectedYear}</span>
            <div className="flex items-center gap-1">
              <Clock className="size-3" />
              <span>{MIN_YEAR} - {MAX_YEAR}</span>
            </div>
          </div>
          <Slider
            value={[selectedYear]}
            onValueChange={(value) => onYearChange(value[0])}
            min={MIN_YEAR}
            max={MAX_YEAR}
            step={1}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>{MIN_YEAR}</span>
            <span className="text-primary font-medium">Current</span>
            <span>{MAX_YEAR}</span>
          </div>
        </div>
        
        {/* Month Slider */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs text-muted-foreground">
            <span>Month: {MONTHS[selectedMonth - 1]}</span>
            <span>1 - 12</span>
          </div>
          <Slider
            value={[selectedMonth]}
            onValueChange={(value) => onMonthChange(value[0])}
            min={1}
            max={12}
            step={1}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground">
            <span>Jan</span>
            <span>Apr</span>
            <span>Jul</span>
            <span>Oct</span>
            <span>Dec</span>
          </div>
        </div>
        
        {/* Current Selection Display */}
        <div className="text-center text-sm bg-muted/50 rounded-md p-3">
          <div className="font-medium">
            Showing predictions for {MONTHS[selectedMonth - 1]} {selectedYear}
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {selectedYear === 2024 ? 'Historical data' : 'AI/ML predictions based on climate models'}
          </div>
        </div>
        
        {/* Animation Controls Info */}
        <div className="text-xs text-muted-foreground text-center">
          <div className="flex items-center justify-center gap-4">
            <span>• Zones will animate smoothly between time periods</span>
            <span>• Use Play to auto-advance through months</span>
            <span>• Hover zones for detailed information</span>
          </div>
        </div>
      </div>
    </div>
  );
}
