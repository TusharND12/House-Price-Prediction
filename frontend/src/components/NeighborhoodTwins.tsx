import { useState } from 'react'
import { Users } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function NeighborhoodTwins({ dark, inputs: _inputs }: Props) {
  const [lat, setLat] = useState(36.0)
  const [lon, setLon] = useState(-119.0)
  const [twins, setTwins] = useState<{ latitude: number; longitude: number; median_house_value: number }[]>([])
  const [message, setMessage] = useState('')

  const fetchTwins = () => {
    fetch(`${API}/neighborhood-twins?lat=${lat}&lon=${lon}`)
      .then(r => r.json())
      .then(res => {
        setTwins(res.twins || [])
        setMessage(res.message || '')
      })
  }

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Users size={20} /> Neighborhood Twins
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Find similar neighborhoods by location. (California Housing data)
      </p>
      <div className="grid grid-cols-2 gap-2">
        <div>
          <label className="block text-xs mb-1">Latitude</label>
          <input
            type="number"
            step={0.1}
            value={lat}
            onChange={e => setLat(Number(e.target.value))}
            className="w-full px-2 py-1 rounded border dark:bg-slate-800"
          />
        </div>
        <div>
          <label className="block text-xs mb-1">Longitude</label>
          <input
            type="number"
            step={0.1}
            value={lon}
            onChange={e => setLon(Number(e.target.value))}
            className="w-full px-2 py-1 rounded border dark:bg-slate-800"
          />
        </div>
      </div>
      <button
        onClick={fetchTwins}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
      >
        Find Twins
      </button>
      {message && <p className="text-sm text-amber-600">{message}</p>}
      {twins.length > 0 && (
        <div className="space-y-2">
          {twins.map((t, i) => (
            <div key={i} className="p-2 rounded bg-slate-100 dark:bg-slate-700">
              <p className="text-sm">({t.latitude?.toFixed(2)}, {t.longitude?.toFixed(2)})</p>
              <p className="font-bold text-indigo-600">${t.median_house_value?.toLocaleString()}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
