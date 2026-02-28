import { useState, useEffect, useMemo } from 'react'
import { MapContainer, TileLayer, CircleMarker, Popup } from 'react-leaflet'

const API = '/api'

function MapContent({ data }: { data: { longitude: number; latitude: number; median_house_value: number }[] }) {
  if (!data.length) return null
  const maxVal = Math.max(...data.map((d) => d.median_house_value))
  return (
    <>
      <TileLayer
        url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        attribution='&copy; OpenStreetMap'
      />
      {data.slice(0, 200).map((d, i) => (
        <CircleMarker
          key={i}
          center={[d.latitude, d.longitude]}
          radius={4 + (d.median_house_value / maxVal) * 6}
          pathOptions={{ color: '#6366f1', fillColor: '#6366f1', fillOpacity: 0.6, weight: 1 }}
        >
          <Popup>
            <strong>Value:</strong> ${d.median_house_value?.toLocaleString() || 0}
          </Popup>
        </CircleMarker>
      ))}
    </>
  )
}

export default function MapView() {
  const [data, setData] = useState<{ longitude: number; latitude: number; median_house_value: number }[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetch(`${API}/map-data`)
      .then((r) => r.json())
      .then((res) => {
        setData(res.data || [])
      })
      .finally(() => setLoading(false))
  }, [])

  const center: [number, number] = useMemo(() => {
    if (!data.length) return [36.5, -119.5]
    const lon = data.reduce((a, d) => a + d.longitude, 0) / data.length
    const lat = data.reduce((a, d) => a + d.latitude, 0) / data.length
    return [lat, lon]
  }, [data])

  if (loading) return <p className="text-slate-500">Loading map...</p>
  if (!data.length) {
    return (
      <div className="bg-white dark:bg-slate-800 rounded-xl p-12 text-center border border-slate-200 dark:border-slate-700">
        <p className="text-slate-600 dark:text-slate-400">
          Map data available for California Housing dataset. Add housing.csv to data/ and use that dataset.
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-white">Map Explorer</h2>
      <p className="text-slate-600 dark:text-slate-400">
        Click on markers to see median house values by location. (California Housing dataset)
      </p>
      <div className="h-[500px] rounded-xl overflow-hidden border border-slate-200 dark:border-slate-700">
        <MapContainer center={center} zoom={6} style={{ height: '100%', width: '100%' }}>
          <MapContent data={data} />
        </MapContainer>
      </div>
    </div>
  )
}
