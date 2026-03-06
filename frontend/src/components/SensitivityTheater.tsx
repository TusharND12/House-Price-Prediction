import { useState, useEffect } from 'react'
import { Zap } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function SensitivityTheater({ dark, inputs }: Props) {
  const [cascades, setCascades] = useState<{ feature: string; change: string; old_prediction: number; new_prediction: number; delta: number }[]>([])
  const [prediction, setPrediction] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchCascades = () => {
    setLoading(true)
    setError(null)
    fetch(`${API}/sensitivity`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then(res => {
        if (res.error) {
          setError(res.error)
          setCascades([])
          setPrediction(null)
        } else {
          setCascades(res.cascades || [])
          setPrediction(res.prediction)
        }
      })
      .catch(() => setError('Could not run sensitivity. Please try again.'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchCascades() }, [inputs])

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Zap size={20} /> Sensitivity Theater
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        See the cascade effect when one feature changes—how much does the price shift?
      </p>
      <button
        onClick={fetchCascades}
        disabled={loading}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {loading ? 'Running…' : 'Run Sensitivity'}
      </button>
      {error && <p className="text-xs text-rose-500">{error}</p>}
      {prediction !== null && (
        <>
          <p className="text-lg font-bold text-indigo-600">Base: ${prediction.toLocaleString()}</p>
          <div className="space-y-3">
            {cascades.map((c, i) => (
              <div
                key={i}
                className="p-4 rounded-lg bg-slate-100 dark:bg-slate-700 border-l-4 border-indigo-500"
              >
                <p className="font-medium">{c.feature.replace(/_/g, ' ')} {c.change}</p>
                <p className="text-sm text-slate-600 dark:text-slate-400">
                  ${c.old_prediction.toLocaleString()} → ${c.new_prediction.toLocaleString()}
                </p>
                <p className={`font-bold ${c.delta >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  {c.delta >= 0 ? '+' : ''}{c.delta.toLocaleString()}
                </p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
