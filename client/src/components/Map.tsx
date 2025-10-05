'use client';

import { useEffect, useRef, useState } from 'react';
import Map, { MapRef, Source, Layer, Popup } from 'react-map-gl/mapbox';
import 'mapbox-gl/dist/mapbox-gl.css';
import { Fish, Thermometer, MapPin, Info, Activity, Zap } from 'lucide-react';
import { useRealTimeTracking } from './RealTimeTracker';

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

interface MapboxMapProps {
  className?: string;
  selectedShark?: Shark | null;
  selectedYear?: number;
  selectedMonth?: number;
}


// Map default location: San Francisco
const DEFAULT_VIEW_STATE = {
  longitude: -122.4194,
  latitude: 37.7749,
  zoom: 8,
} as const;

const MAPBOX_TOKEN = process.env.NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN;

/**
 * Interactive Mapbox map component
 * Shows shark zones based on selected species
 */
export default function MapboxMap({ className, selectedShark, selectedYear = 2024, selectedMonth = 6 }: MapboxMapProps) {
  const mapRef = useRef<MapRef>(null);
  const [hoveredZone, setHoveredZone] = useState<SharkZone | null>(null);
  const [clickedZone, setClickedZone] = useState<SharkZone | null>(null);
  const [isAnimating, setIsAnimating] = useState(false);
  const [hoveredTracker, setHoveredTracker] = useState<any>(null);
  const [clickedTracker, setClickedTracker] = useState<any>(null);
  
  // Real-time tracking
  const { 
    trackingData, 
    isConnected, 
    lastPing, 
    getTrackingGeoJSON, 
    getJawMotionColor, 
    getPulsationIntensity 
  } = useRealTimeTracking(selectedShark?.id);
  
  // Convert shark zones to GeoJSON format based on time
  const getSharkZonesGeoJSON = () => {
    if (!selectedShark || !selectedShark.zones) return { type: "FeatureCollection" as const, features: [] };
    
    // Get zones for the selected year and month
    const yearData = selectedShark.zones[selectedYear.toString()];
    if (!yearData) return { type: "FeatureCollection" as const, features: [] };
    
    const monthZones = yearData[selectedMonth.toString()];
    if (!monthZones) return { type: "FeatureCollection" as const, features: [] };
    
    const features = monthZones.map(zone => ({
      type: "Feature" as const,
      properties: {
        id: zone.id,
        name: zone.name,
        sst: zone.sst,
        intensity: zone.intensity,
        description: zone.description,
        sharkName: selectedShark.name,
        sharkScientificName: selectedShark.scientificName,
        year: selectedYear,
        month: selectedMonth
      },
      geometry: {
        type: "Polygon" as const,
        coordinates: [zone.coordinates]
      }
    }));
    
    return {
      type: "FeatureCollection" as const,
      features
    };
  };
  
  const currentSharkData = getSharkZonesGeoJSON();
  
  // Animation effect when time changes
  useEffect(() => {
    if (selectedShark) {
      setIsAnimating(true);
      const timer = setTimeout(() => {
        setIsAnimating(false);
      }, 1000); // Animation duration
      
      return () => clearTimeout(timer);
    }
  }, [selectedYear, selectedMonth, selectedShark]);


  // Function to move/center the map on specific coordinates
  const moveToLocation = (latitude: number, longitude: number, zoom: number = 14) => {
    if (mapRef.current) {
      mapRef.current.flyTo({
        center: [longitude, latitude],
        zoom,
        duration: 2000, // 2 second smooth transition
      });
    }
  };

  // Get intensity color for zones
  const getIntensityColor = (intensity: number) => {
    if (intensity >= 0.8) return '#ef4444'; // red
    if (intensity >= 0.6) return '#f97316'; // orange
    if (intensity >= 0.4) return '#eab308'; // yellow
    return '#22c55e'; // green
  };

  // Get intensity opacity for zones
  const getIntensityOpacity = (intensity: number) => {
    return Math.max(0.3, intensity);
  };

  // Expose functions globally for AI integration
  useEffect(() => {
    if (typeof window !== 'undefined') {
      const windowWithFunctions = window as typeof window & {
        moveToLocation?: (lat: number, lng: number, zoom?: number) => void;
      };
      
      windowWithFunctions.moveToLocation = moveToLocation;
      
      return () => {
        delete windowWithFunctions.moveToLocation;
      };
    }
  }, []);


  if (!MAPBOX_TOKEN) {
    return (
      <div className={className}>
        <p className="text-sm text-muted-foreground">
          Map unavailable: missing Mapbox access token.
        </p>
      </div>
    );
  }

  return (
    <div className={className}>
      <Map
        ref={mapRef}
        initialViewState={DEFAULT_VIEW_STATE}
        mapboxAccessToken={MAPBOX_TOKEN}
        mapStyle="mapbox://styles/mapbox/streets-v12"
        style={{ width: '100%', height: '100%' }}
        interactiveLayerIds={['shark-zones', 'tracking-sharks', 'tracking-pulse']}
        onMouseEnter={(e) => {
          if (e.features && e.features.length > 0) {
            const feature = e.features[0];
            if (feature.layer.id === 'tracking-sharks' || feature.layer.id === 'tracking-pulse') {
              setHoveredTracker({
                id: feature.properties.id,
                name: feature.properties.name,
                jawMotion: feature.properties.jawMotion,
                jawTrend: feature.properties.jawTrend,
                lastUpdate: feature.properties.lastUpdate,
                status: feature.properties.status
              });
            } else {
              setHoveredZone({
                id: feature.properties.id,
                name: feature.properties.name,
                coordinates: feature.geometry.coordinates[0],
                sst: feature.properties.sst,
                intensity: feature.properties.intensity,
                description: feature.properties.description
              });
            }
          }
        }}
        onMouseLeave={() => {
          setHoveredZone(null);
          setHoveredTracker(null);
        }}
        onClick={(e) => {
          if (e.features && e.features.length > 0) {
            const feature = e.features[0];
            if (feature.layer.id === 'tracking-sharks' || feature.layer.id === 'tracking-pulse') {
              setClickedTracker({
                id: feature.properties.id,
                name: feature.properties.name,
                jawMotion: feature.properties.jawMotion,
                jawTrend: feature.properties.jawTrend,
                lastUpdate: feature.properties.lastUpdate,
                status: feature.properties.status
              });
            } else {
              setClickedZone({
                id: feature.properties.id,
                name: feature.properties.name,
                coordinates: feature.geometry.coordinates[0],
                sst: feature.properties.sst,
                intensity: feature.properties.intensity,
                description: feature.properties.description
              });
            }
          }
        }}
      >
        {/* Shark Zones */}
        <Source id="shark-zones" type="geojson" data={currentSharkData}>
          <Layer
            id="shark-zones"
            type="fill"
            paint={{
              'fill-color': [
                'case',
                ['boolean', ['feature-state', 'hover'], false],
                '#ffffff',
                [
                  'interpolate',
                  ['linear'],
                  ['get', 'intensity'],
                  0, '#22c55e',
                  0.4, '#eab308',
                  0.6, '#f97316',
                  0.8, '#ef4444',
                  1, '#dc2626'
                ]
              ],
              'fill-opacity': [
                'case',
                ['boolean', ['feature-state', 'hover'], false],
                0.8,
                [
                  'interpolate',
                  ['linear'],
                  ['get', 'intensity'],
                  0, 0.3,
                  1, 0.7
                ]
              ]
            }}
            layout={{
              'fill-sort-key': ['get', 'intensity']
            }}
          />
          <Layer
            id="shark-zones-outline"
            type="line"
            paint={{
              'line-color': [
                'interpolate',
                ['linear'],
                ['get', 'intensity'],
                0, '#16a34a',
                0.4, '#ca8a04',
                0.6, '#ea580c',
                0.8, '#dc2626',
                1, '#991b1b'
              ],
              'line-width': [
                'case',
                ['boolean', ['feature-state', 'hover'], false],
                3,
                2
              ],
              'line-opacity': [
                'case',
                ['boolean', ['feature-state', 'hover'], false],
                1,
                0.8
              ]
            }}
          />
        </Source>

        {/* Real-time Tracking Data */}
        <Source id="tracking-sharks" type="geojson" data={getTrackingGeoJSON()}>
          {/* Pulsating circles for jaw motion intensity */}
          <Layer
            id="tracking-pulse"
            type="circle"
            paint={{
              'circle-radius': [
                'interpolate',
                ['linear'],
                ['get', 'pulsationIntensity'],
                0, 20,
                1, 60
              ],
              'circle-color': [
                'interpolate',
                ['linear'],
                ['get', 'jawMotion'],
                0, '#16a34a',
                0.4, '#ca8a04',
                0.6, '#ea580c',
                0.8, '#dc2626',
                1, '#991b1b'
              ],
              'circle-opacity': [
                'interpolate',
                ['linear'],
                ['get', 'pulsationIntensity'],
                0, 0.1,
                1, 0.3
              ],
              'circle-stroke-width': 0
            }}
          />
          
          {/* Main tracking points */}
          <Layer
            id="tracking-sharks"
            type="circle"
            paint={{
              'circle-radius': [
                'interpolate',
                ['linear'],
                ['get', 'jawMotion'],
                0, 8,
                1, 16
              ],
              'circle-color': [
                'interpolate',
                ['linear'],
                ['get', 'jawMotion'],
                0, '#16a34a',
                0.4, '#ca8a04',
                0.6, '#ea580c',
                0.8, '#dc2626',
                1, '#991b1b'
              ],
              'circle-stroke-color': 'white',
              'circle-stroke-width': 2,
              'circle-opacity': 0.9
            }}
          />
        </Source>
      </Map>

      {/* Hover Popup for Zones */}
      {hoveredZone && (
        <div className="absolute top-4 right-4 bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 text-sm max-w-[250px] z-10">
          <div className="flex items-center gap-2 mb-2">
            <Fish className="size-4" />
            <span className="font-semibold">{hoveredZone.name}</span>
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <Thermometer className="size-3" />
              <span>SST: {hoveredZone.sst}°C</span>
            </div>
            <div className="flex items-center gap-2">
              <MapPin className="size-3" />
              <span>Intensity: {(hoveredZone.intensity * 100).toFixed(0)}%</span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              {hoveredZone.description}
            </div>
          </div>
        </div>
      )}

      {/* Hover Popup for Tracking Data */}
      {hoveredTracker && (
        <div className="absolute top-4 right-4 bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 text-sm max-w-[250px] z-10">
          <div className="flex items-center gap-2 mb-2">
            <Activity className="size-4" />
            <span className="font-semibold">{hoveredTracker.name}</span>
            <div className={`w-2 h-2 rounded-full ${
              hoveredTracker.status === 'active' ? 'bg-green-500' : 'bg-red-500'
            }`}></div>
          </div>
          <div className="space-y-1 text-xs">
            <div className="flex items-center gap-2">
              <Zap className="size-3" />
              <span>Jaw Motion: {(hoveredTracker.jawMotion * 100).toFixed(0)}%</span>
            </div>
            <div className="flex items-center gap-2">
              <span className={`px-2 py-1 rounded text-xs ${
                hoveredTracker.jawTrend === 'increasing' ? 'bg-red-100 text-red-800' :
                hoveredTracker.jawTrend === 'decreasing' ? 'bg-blue-100 text-blue-800' :
                'bg-gray-100 text-gray-800'
              }`}>
                {hoveredTracker.jawTrend === 'increasing' ? '↗ Increasing' :
                 hoveredTracker.jawTrend === 'decreasing' ? '↘ Decreasing' :
                 '→ Stable'}
              </span>
            </div>
            <div className="text-xs text-muted-foreground mt-2">
              Last update: {new Date(hoveredTracker.lastUpdate).toLocaleTimeString()}
            </div>
          </div>
        </div>
      )}

      {/* Click Popup for Zones */}
      {clickedZone && (
        <Popup
          longitude={clickedZone.coordinates[0][0]}
          latitude={clickedZone.coordinates[0][1]}
          onClose={() => setClickedZone(null)}
          closeButton={true}
          closeOnClick={false}
          className="shark-zone-popup"
        >
          <div className="p-2 min-w-[200px]">
            <div className="flex items-center gap-2 mb-2">
              <Fish className="size-4" />
              <span className="font-semibold">{clickedZone.name}</span>
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <Thermometer className="size-3" />
                <span>SST: {clickedZone.sst}°C</span>
              </div>
              <div className="flex items-center gap-2">
                <MapPin className="size-3" />
                <span>Intensity: {(clickedZone.intensity * 100).toFixed(0)}%</span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                {clickedZone.description}
              </div>
            </div>
          </div>
        </Popup>
      )}

      {/* Click Popup for Tracking Data */}
      {clickedTracker && (
        <Popup
          longitude={trackingData.find(t => t.id === clickedTracker.id)?.currentPosition.lng || 0}
          latitude={trackingData.find(t => t.id === clickedTracker.id)?.currentPosition.lat || 0}
          onClose={() => setClickedTracker(null)}
          closeButton={true}
          closeOnClick={false}
          className="tracking-popup"
        >
          <div className="p-2 min-w-[200px]">
            <div className="flex items-center gap-2 mb-2">
              <Activity className="size-4" />
              <span className="font-semibold">{clickedTracker.name}</span>
              <div className={`w-2 h-2 rounded-full ${
                clickedTracker.status === 'active' ? 'bg-green-500' : 'bg-red-500'
              }`}></div>
            </div>
            <div className="space-y-1 text-xs">
              <div className="flex items-center gap-2">
                <Zap className="size-3" />
                <span>Jaw Motion: {(clickedTracker.jawMotion * 100).toFixed(0)}%</span>
              </div>
              <div className="flex items-center gap-2">
                <span className={`px-2 py-1 rounded text-xs ${
                  clickedTracker.jawTrend === 'increasing' ? 'bg-red-100 text-red-800' :
                  clickedTracker.jawTrend === 'decreasing' ? 'bg-blue-100 text-blue-800' :
                  'bg-gray-100 text-gray-800'
                }`}>
                  {clickedTracker.jawTrend === 'increasing' ? '↗ Increasing' :
                   clickedTracker.jawTrend === 'decreasing' ? '↘ Decreasing' :
                   '→ Stable'}
                </span>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Last update: {new Date(clickedTracker.lastUpdate).toLocaleTimeString()}
              </div>
            </div>
          </div>
        </Popup>
      )}

      {/* Legend */}
      {selectedShark && (
        <div className="absolute bottom-4 left-4 bg-card/95 backdrop-blur-sm border border-border rounded-lg p-3 text-sm max-w-[320px]">
          <h3 className="font-semibold mb-2 flex items-center gap-2">
            <Fish className="size-4" />
            {selectedShark.name} Zones
          </h3>
          <div className="space-y-2">
            {/* Time info */}
            <div className="text-xs text-primary font-medium">
              {new Date(selectedYear, selectedMonth - 1).toLocaleDateString('en-US', { 
                month: 'long', 
                year: 'numeric' 
              })}
              {isAnimating && <span className="ml-2 animate-pulse">🔄</span>}
            </div>
            
            {/* Real-time tracking status */}
            {isConnected && (
              <div className="flex items-center gap-2 text-xs">
                <div className="flex items-center gap-1">
                  <Activity className="size-3 text-green-500" />
                  <span className="text-green-600 font-medium">Live Tracking</span>
                </div>
                <div className="text-muted-foreground">
                  {trackingData.length} sharks • Last ping: {lastPing?.toLocaleTimeString()}
                </div>
              </div>
            )}
            
            {/* Intensity color gradient */}
            <div className="flex items-center gap-2">
              <div className="flex h-3 w-16 rounded overflow-hidden">
                <div className="w-1/4 bg-green-500/60"></div>
                <div className="w-1/4 bg-yellow-500/60"></div>
                <div className="w-1/4 bg-orange-500/60"></div>
                <div className="w-1/4 bg-red-500/60"></div>
              </div>
              <span className="text-xs">Low → High Risk</span>
            </div>
            
            {/* Zone count */}
            <div className="text-xs text-muted-foreground">
              {currentSharkData.features.length} zones • SST: {selectedShark.optimalSST.min}°C - {selectedShark.optimalSST.max}°C
            </div>
            
            {/* Real-time tracking legend */}
            {isConnected && (
              <div className="text-xs text-muted-foreground mt-2 pt-2 border-t border-border">
                <div className="font-medium mb-1">Real-time Tracking:</div>
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                    <span>Pulsating zones show jaw motion intensity</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-red-500"></div>
                    <span>Red = High activity, Green = Low activity</span>
                  </div>
                </div>
              </div>
            )}
            
            <div className="text-xs text-muted-foreground mt-2 pt-2 border-t border-border">
              Click zones for details • Hover for quick info
            </div>
          </div>
        </div>
      )}

      {/* No Selection Message */}
      {!selectedShark && (
        <div className="absolute inset-0 flex items-center justify-center bg-background/50 backdrop-blur-sm">
          <div className="bg-card/95 backdrop-blur-sm border border-border rounded-lg p-6 text-center max-w-[300px]">
            <Fish className="size-12 mx-auto text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">Select a Shark Species</h3>
            <p className="text-sm text-muted-foreground">
              Choose a shark species from the sidebar to view their optimal SST zones and activity areas.
            </p>
          </div>
        </div>
      )}
    </div>
  );
}
