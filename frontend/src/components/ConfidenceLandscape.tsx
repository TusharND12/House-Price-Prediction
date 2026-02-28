import { useState, useEffect } from 'react'
import { Gauge } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function ConfidenceLandscape({ dark, inputs }: Props) {
  const [confidence, setConfidence] = useState<number | null>(null)
  const [prediction, setPrediction] = useState<number | null>(null)
  const [interpretation, setInterpretation] = useState('')

  const fetchConfidence = () => {
    fetch(`${API}/confidence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then(res => {
        if (!res.error) {
          setConfidence(res.confidence)
          setPrediction(res.prediction)
          setInterpretation(res.interpretation || '')
        }
      })
  }

  useEffect(() => { fetchConfidence() }, [])

  const pct = confidence !== null ? confidence * 100 : 0
  const color = interpretation === 'high' ? '#22c55e' : interpretation === 'medium' ? '#f59e0b' : '#ef4444'

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Gauge size={20} /> Confidence Landscape
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        How confident is the model? High = typical inputs, Low = unusual.
      </p>
      <button
        onClick={fetchConfidence}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
      >
        Check Confidence
      </button>
      {confidence !== null && (
        <>
          <p className="text-lg font-bold text-indigo-600">${prediction?.toLocaleString()}</p>
          <div className="relative h-8 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden">
            <div
              className="h-full rounded-full transition-all duration-500"
              style={{ width: `${pct}%`, backgroundColor: color }}
            />
          </div>
          <p className="text-center font-medium capitalize" style={{ color }}>{interpretation} confidence</p>
        </>
      )}
    </div>
  )
}
