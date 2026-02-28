import { useState } from 'react'
import { CloudRain } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function ClimateRiskPanel({ dark, inputs: _inputs }: Props) {
  const [lat, setLat] = useState(36.0)
  const [lon, setLon] = useState(-119.0)
  const [risk, setRisk] = useState<{ flood_risk: number; fire_risk: number } | null>(null)

  const fetchRisk = () => {
    fetch(`${API}/climate-risk?lat=${lat}&lon=${lon}`)
      .then(r => r.json())
      .then(res => {
        setRisk({ flood_risk: res.flood_risk, fire_risk: res.fire_risk })
      })
  }

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <CloudRain size={20} /> Climate Risk Overlay
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Synthetic flood & fire risk by location (California). Enter coordinates.
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
        onClick={fetchRisk}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
      >
        Get Risk
      </button>
      {risk && (
        <div className="space-y-3">
          <div>
            <p className="text-sm">Flood Risk</p>
            <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-blue-500 rounded-full"
                style={{ width: `${risk.flood_risk * 100}%` }}
              />
            </div>
            <p className="text-xs">{(risk.flood_risk * 100).toFixed(0)}%</p>
          </div>
          <div>
            <p className="text-sm">Fire Risk</p>
            <div className="h-3 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
              <div
                className="h-full bg-orange-500 rounded-full"
                style={{ width: `${risk.fire_risk * 100}%` }}
              />
            </div>
            <p className="text-xs">{(risk.fire_risk * 100).toFixed(0)}%</p>
          </div>
        </div>
      )}
    </div>
  )
}
