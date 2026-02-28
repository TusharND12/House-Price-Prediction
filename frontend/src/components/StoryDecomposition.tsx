import { useState, useEffect } from 'react'
import { PieChart, Pie, Cell, ResponsiveContainer, Legend, Tooltip } from 'recharts'
import type { PlaygroundInputs } from '../App'

const API = '/api'

const STORY_COLORS: Record<string, string> = {
  structure: '#6366f1',
  quality: '#22c55e',
  lifestyle: '#f59e0b',
  location: '#06b6d4',
}

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function StoryDecomposition({ dark, inputs }: Props) {
  const [stories, setStories] = useState<{ category: string; total: number; items: { name: string; contribution: number }[] }[]>([])
  const [prediction, setPrediction] = useState<number | null>(null)

  const fetchStories = () => {
    fetch(`${API}/story-decomposition`, {
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

  useEffect(() => { fetchStories() }, [inputs])

  const pieData = stories.map(s => ({
    name: s.category.charAt(0).toUpperCase() + s.category.slice(1),
    value: Math.abs(s.total),
    color: STORY_COLORS[s.category] || '#94a3b8',
  })).filter(d => d.value > 0)

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white">Story-Based Price Decomposition</h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        The price broken into narratives (from Playground values).
      </p>
      <button onClick={fetchStories} className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg">Update</button>
      {prediction !== null && (
        <>
          <p className="text-2xl font-bold text-indigo-600 dark:text-indigo-400">${prediction.toLocaleString()}</p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={70} label>
                  {pieData.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Pie>
                <Tooltip formatter={(v: number) => `$${v.toLocaleString()}`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="space-y-2">
            {stories.map(s => (
              <div key={s.category} className="p-2 rounded bg-slate-100 dark:bg-slate-700">
                <span className="font-medium capitalize">{s.category}:</span> ${s.total.toLocaleString()}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
