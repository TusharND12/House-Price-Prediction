import { useState, useEffect } from 'react'
import { Baby } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function SimpleExplain({ dark, inputs }: Props) {
  const [simple, setSimple] = useState('')
  const [prediction, setPrediction] = useState<number | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetchSimple = () => {
    setLoading(true)
    setError(null)
    fetch(`${API}/simple-explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then(res => {
        if (res.error) {
          setError(res.error)
          setSimple('')
          setPrediction(null)
        } else {
          setSimple(res.simple)
          setPrediction(res.prediction)
        }
      })
      .catch(() => setError('Unable to fetch explanation. Please try again.'))
      .finally(() => setLoading(false))
  }

  useEffect(() => { fetchSimple() }, [inputs])

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Baby size={20} /> Explainability for Everyone
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Kid-friendly analogies: &quot;like a bigger toy box&quot;—no jargon.
      </p>
      <button
        onClick={fetchSimple}
        disabled={loading}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {loading ? 'Loading…' : 'Get Simple Explanation'}
      </button>
      {error && <p className="text-xs text-rose-500 mt-1">{error}</p>}
      {simple && (
        <div className="p-4 rounded-lg bg-amber-50 dark:bg-amber-900/20 border-2 border-amber-300 dark:border-amber-700">
          {prediction && <p className="text-lg font-bold text-indigo-600 mb-2">${prediction.toLocaleString()}</p>}
          <p className="text-lg text-slate-700 dark:text-slate-200 italic">&quot;{simple}&quot;</p>
        </div>
      )}
    </div>
  )
}
