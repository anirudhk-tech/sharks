'use client';

import { useState, useEffect, useRef } from 'react';
import { Activity, Zap, AlertTriangle } from 'lucide-react';

interface TrackingShark {
  id: string;
  sharkId: string;
  name: string;
  currentPosition: {
    lat: number;
    lng: number;
  };
  jawMotion: {
    current: number;
    previous: number;
    trend: 'increasing' | 'decreasing' | 'stable';
  };
  lastUpdate: string;
  status: 'active' | 'inactive';
}

interface RealTimeTrackerProps {
  selectedShark?: string | null;
  onTrackingDataUpdate?: (data: TrackingShark[]) => void;
}

export default function RealTimeTracker({ selectedShark, onTrackingDataUpdate }: RealTimeTrackerProps) {
  const [trackingData, setTrackingData] = useState<TrackingShark[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const [lastPing, setLastPing] = useState<Date | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  // Fetch real-time data updates every 5 seconds
  useEffect(() => {
    const fetchTrackingData = async () => {
      try {
        const url = selectedShark 
          ? `/api/tracking?sharkId=${selectedShark}`
          : '/api/tracking';
        
        const response = await fetch(url);
        if (!response.ok) throw new Error('Failed to fetch tracking data');
        
        const data = await response.json();
        const updatedData = data.sharks.map((shark: any) => {
          const prevShark = trackingData.find(p => p.id === shark.id);
          if (prevShark) {
            // Calculate trend based on previous data
            const trend = shark.jawMotion.current > prevShark.jawMotion.current 
              ? 'increasing' 
              : shark.jawMotion.current < prevShark.jawMotion.current 
                ? 'decreasing' 
                : 'stable';
            
            return {
              ...shark,
              jawMotion: {
                ...shark.jawMotion,
                previous: prevShark.jawMotion.current,
                trend
              }
            };
          }
          return shark;
        });

        setTrackingData(updatedData);
        onTrackingDataUpdate?.(updatedData);
        setLastPing(new Date());
      } catch (error) {
        console.error('Error fetching tracking data:', error);
        setIsConnected(false);
      }
    };

    // Start fetching
    setIsConnected(true);
    fetchTrackingData(); // Initial data
    intervalRef.current = setInterval(fetchTrackingData, 5000);

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      setIsConnected(false);
    };
  }, [selectedShark, onTrackingDataUpdate, trackingData]);

  // Get color intensity based on jaw motion trend
  const getJawMotionColor = (jawMotion: TrackingShark['jawMotion']) => {
    const baseIntensity = jawMotion.current;
    const trendMultiplier = jawMotion.trend === 'increasing' ? 1.3 : 
                           jawMotion.trend === 'decreasing' ? 0.7 : 1.0;
    
    const intensity = Math.min(1, baseIntensity * trendMultiplier);
    
    if (intensity >= 0.8) return '#dc2626'; // red
    if (intensity >= 0.6) return '#ea580c'; // orange
    if (intensity >= 0.4) return '#ca8a04'; // yellow
    return '#16a34a'; // green
  };

  // Get pulsation intensity
  const getPulsationIntensity = (jawMotion: TrackingShark['jawMotion']) => {
    const change = Math.abs(jawMotion.current - jawMotion.previous);
    return Math.max(0.3, Math.min(1, change * 2));
  };

  // Convert tracking data to GeoJSON for map display
  const getTrackingGeoJSON = () => {
    const features = trackingData.map(shark => ({
      type: "Feature" as const,
      properties: {
        id: shark.id,
        name: shark.name,
        sharkId: shark.sharkId,
        jawMotion: shark.jawMotion.current,
        jawTrend: shark.jawMotion.trend,
        lastUpdate: shark.lastUpdate,
        status: shark.status,
        color: getJawMotionColor(shark.jawMotion),
        pulsationIntensity: getPulsationIntensity(shark.jawMotion)
      },
      geometry: {
        type: "Point" as const,
        coordinates: [shark.currentPosition.lng, shark.currentPosition.lat]
      }
    }));

    return {
      type: "FeatureCollection" as const,
      features
    };
  };

  return {
    trackingData,
    isConnected,
    lastPing,
    getTrackingGeoJSON,
    getJawMotionColor,
    getPulsationIntensity
  };
}

// Hook for using real-time tracking data
export function useRealTimeTracking(selectedShark?: string | null) {
  const [trackingData, setTrackingData] = useState<TrackingShark[]>([]);
  
  const tracker = RealTimeTracker({
    selectedShark,
    onTrackingDataUpdate: setTrackingData
  });

  return {
    ...tracker,
    trackingData
  };
}
