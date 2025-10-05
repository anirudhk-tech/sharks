'use client';

import { useState, useEffect } from 'react';
import { ChevronDown, Fish, Thermometer, MapPin, Info, Loader2 } from 'lucide-react';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';

interface SharkZone {
  id: string;
  name: string;
  coordinates: number[][];
  sst: number;
  intensity: number;
  description: string;
}

interface Shark {
  id: string;
  name: string;
  scientificName: string;
  optimalSST: {
    min: number;
    max: number;
    preferred: number;
  };
  zones: SharkZone[];
}

interface SharkSidebarProps {
  selectedShark: Shark | null;
  onSharkSelect: (shark: Shark | null) => void;
  isOpen: boolean;
  onToggle: () => void;
}

export default function SharkSidebar({ selectedShark, onSharkSelect, isOpen, onToggle }: SharkSidebarProps) {
  const [expandedShark, setExpandedShark] = useState<string | null>(null);
  const [sharkData, setSharkData] = useState<{ sharks: Shark[] } | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch shark data from API
  useEffect(() => {
    const fetchSharkData = async () => {
      try {
        const response = await fetch('/api/sharks');
        if (!response.ok) throw new Error('Failed to fetch shark data');
        const data = await response.json();
        setSharkData(data);
      } catch (error) {
        console.error('Error fetching shark data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSharkData();
  }, []);

  const handleSharkSelect = (shark: Shark) => {
    onSharkSelect(shark);
    setExpandedShark(null);
  };

  const getIntensityColor = (intensity: number) => {
    if (intensity >= 0.8) return 'bg-red-500';
    if (intensity >= 0.6) return 'bg-orange-500';
    if (intensity >= 0.4) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  const getIntensityLabel = (intensity: number) => {
    if (intensity >= 0.8) return 'High Risk';
    if (intensity >= 0.6) return 'Moderate Risk';
    if (intensity >= 0.4) return 'Low Risk';
    return 'Safe';
  };

  if (loading) {
    return (
      <div className={`absolute top-0 left-0 h-full bg-card border-r border-border z-20 transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'} w-80`}>
        <div className="p-4 h-full flex items-center justify-center">
          <div className="text-center">
            <Loader2 className="size-8 animate-spin mx-auto mb-2" />
            <p className="text-sm text-muted-foreground">Loading shark data...</p>
          </div>
        </div>
      </div>
    );
  }

  if (!sharkData) {
    return (
      <div className={`absolute top-0 left-0 h-full bg-card border-r border-border z-20 transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'} w-80`}>
        <div className="p-4 h-full flex items-center justify-center">
          <div className="text-center">
            <Fish className="size-8 mx-auto mb-2 text-muted-foreground" />
            <p className="text-sm text-muted-foreground">Failed to load shark data</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className={`absolute top-0 left-0 h-full bg-card border-r border-border z-20 transition-transform duration-300 ease-in-out ${isOpen ? 'translate-x-0' : '-translate-x-full'} w-80`}>
      <div className="p-4 h-full overflow-y-auto">
        <div className="flex items-center gap-2 mb-6">
          <Fish className="size-5" />
          <h2 className="text-lg font-semibold">Shark Species</h2>
        </div>
        
        <div className="space-y-3">
          {sharkData.sharks.map((shark) => (
            <Collapsible
              key={shark.id}
              open={expandedShark === shark.id}
              onOpenChange={(open) => setExpandedShark(open ? shark.id : null)}
            >
              <CollapsibleTrigger asChild>
                <Button
                  variant={selectedShark?.id === shark.id ? "default" : "outline"}
                  className="w-full justify-between p-3 h-auto"
                  onClick={() => handleSharkSelect(shark)}
                >
                  <div className="flex items-center gap-2">
                    <Fish className="size-4" />
                    <div className="text-left">
                      <div className="font-medium">{shark.name}</div>
                      <div className="text-xs text-muted-foreground">{shark.scientificName}</div>
                    </div>
                  </div>
                  <ChevronDown className="size-4" />
                </Button>
              </CollapsibleTrigger>
              
              <CollapsibleContent className="mt-2 space-y-2">
                <div className="bg-muted/50 rounded-lg p-3 space-y-2">
                  <div className="flex items-center gap-2 text-sm">
                    <Thermometer className="size-4" />
                    <span className="font-medium">Optimal SST Range</span>
                  </div>
                  <div className="text-xs text-muted-foreground">
                    {shark.optimalSST.min}°C - {shark.optimalSST.max}°C
                    <br />
                    <span className="text-primary">Preferred: {shark.optimalSST.preferred}°C</span>
                  </div>
                  
                  <div className="space-y-2">
                    <div className="text-sm font-medium">
                      Time-Based Zones ({Object.keys(shark.zones).length} years)
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Available years: {Object.keys(shark.zones).join(', ')}
                    </div>
                    <div className="text-xs text-muted-foreground">
                      Each year contains monthly zone data showing seasonal patterns
                    </div>
                    <div className="bg-background rounded p-2 text-xs">
                      <div className="flex items-center gap-2 mb-1">
                        <Info className="size-3" />
                        <span className="font-medium">Interactive Time Slider</span>
                      </div>
                      <div className="text-muted-foreground">
                        Use the time slider at the bottom of the map to see how zones change throughout the year
                      </div>
                    </div>
                  </div>
                </div>
              </CollapsibleContent>
            </Collapsible>
          ))}
        </div>
        
        {selectedShark && (
          <div className="mt-6 p-3 bg-primary/10 rounded-lg border border-primary/20">
            <div className="flex items-center gap-2 mb-2">
              <Info className="size-4 text-primary" />
              <span className="text-sm font-medium text-primary">Selected Species</span>
            </div>
            <div className="text-sm">
              <div className="font-medium">{selectedShark.name}</div>
              <div className="text-xs text-muted-foreground">
                {Object.keys(selectedShark.zones).length} years of data • SST: {selectedShark.optimalSST.min}°C - {selectedShark.optimalSST.max}°C
              </div>
            </div>
            <Button
              variant="outline"
              size="sm"
              className="w-full mt-2"
              onClick={() => onSharkSelect(null)}
            >
              Clear Selection
            </Button>
          </div>
        )}
        
        <div className="mt-6 p-3 bg-muted/30 rounded-lg text-xs text-muted-foreground">
          <div className="font-medium mb-1">Legend</div>
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-red-500"></div>
              <span>High Risk (80%+)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-orange-500"></div>
              <span>Moderate Risk (60-79%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-yellow-500"></div>
              <span>Low Risk (40-59%)</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-green-500"></div>
              <span>Safe (0-39%)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
