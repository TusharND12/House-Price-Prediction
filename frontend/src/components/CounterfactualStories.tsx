import { useState, useEffect } from 'react'
import { BookOpen } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function CounterfactualStories({ dark, inputs }: Props) {
  const [stories, setStories] = useState<{ type: string; feature: string; impact: number; narrative: string }[]>([])
  const [prediction, setPrediction] = useState<number | null>(null)

  const fetchStories = () => {
    fetch(`${API}/counterfactual`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then(res => {
        if (!res.error) {
          setStories(res.stories || [])
          setPrediction(res.prediction)
        }
      })
  }

  useEffect(() => { fetchStories() }, [])

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <BookOpen size={20} /> Counterfactual Stories
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        &quot;What if&quot; narratives from your Playground values.
      </p>
      <button onClick={fetchStories} className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg">Update</button>
      {prediction !== null && (
        <p className="text-lg font-bold text-indigo-600">${prediction.toLocaleString()}</p>
      )}
      <div className="space-y-3">
        {stories.map((s, i) => (
          <div
            key={i}
            className={`p-4 rounded-lg border-l-4 ${
              s.type === 'reduce' ? 'border-green-500 bg-green-50 dark:bg-green-900/20' : 'border-amber-500 bg-amber-50 dark:bg-amber-900/20'
            }`}
          >
            <p className="text-slate-700 dark:text-slate-200">{s.narrative}</p>
          </div>
        ))}
      </div>
    </div>
  )
}
