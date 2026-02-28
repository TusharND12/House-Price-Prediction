import { useState, useEffect } from 'react'
import { Target } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function SacrificeGame({ dark, inputs }: Props) {
  const [options, setOptions] = useState<{ action: string; savings: number; new_value: number }[]>([])
  const [prediction, setPrediction] = useState<number | null>(null)
  const [targetSavings, setTargetSavings] = useState(50000)

  const fetchOptions = () => {
    fetch(`${API}/sacrifice-options`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...inputs, target_savings: targetSavings }),
    })
      .then(r => r.json())
      .then(res => {
        if (!res.error) {
          setOptions(res.options || [])
          setPrediction(res.prediction)
        }
      })
  }

  useEffect(() => { fetchOptions() }, [])

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Target size={20} /> What Would You Sacrifice?
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        To save $50K, would you...? Explore trade-offs.
      </p>
      <div>
        <label className="block text-sm mb-1">Target savings ($)</label>
        <input
          type="number"
          value={targetSavings}
          onChange={e => setTargetSavings(Number(e.target.value) || 50000)}
          className="w-full px-3 py-2 rounded border dark:bg-slate-800 dark:border-slate-600"
        />
      </div>
      <button
        onClick={fetchOptions}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
      >
        Get Options
      </button>
      {prediction !== null && (
        <>
          <p className="text-lg font-bold text-indigo-600">Current: ${prediction.toLocaleString()}</p>
          <div className="space-y-2">
            {options.map((o, i) => (
              <div
                key={i}
                className="p-3 rounded-lg bg-slate-100 dark:bg-slate-700 border border-slate-200 dark:border-slate-600"
              >
                <p className="font-medium">To {o.action}</p>
                <p className="text-green-600 dark:text-green-400 font-bold">Save ${o.savings.toLocaleString()}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
