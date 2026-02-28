import { useState, useEffect } from 'react'
import { Dna } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function DreamHomeDNA({ dark, inputs }: Props) {
  const [strands, setStrands] = useState<{ name: string; contribution: number; segment: number }[]>([])
  const [prediction, setPrediction] = useState<number | null>(null)

  const fetchDNA = () => {
    fetch(`${API}/dna-strand`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then(res => {
        if (!res.error) {
          setStrands(res.strands || [])
          setPrediction(res.prediction)
        }
      })
  }

  useEffect(() => { fetchDNA() }, [])

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Dna size={20} /> Dream Home DNA
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Each segment is a feature—hover to see its contribution. Your house&apos;s &quot;DNA&quot;.
      </p>
      <button
        onClick={fetchDNA}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
      >
        Generate DNA
      </button>
      {prediction !== null && strands.length > 0 && (
        <>
          <p className="text-lg font-bold text-indigo-600">${prediction.toLocaleString()}</p>
          <div className="flex flex-wrap gap-1">
            {strands.map((s, i) => (
              <div
                key={i}
                className="rounded px-2 py-1 text-xs font-medium transition hover:scale-105"
                style={{
                  width: `${s.segment}%`,
                  minWidth: 20,
                  backgroundColor: s.contribution >= 0 ? '#22c55e' : '#ef4444',
                  color: 'white',
                }}
                title={`${s.name}: $${s.contribution.toLocaleString()}`}
              >
                {s.name.replace(/_/g, ' ').slice(0, 8)}
              </div>
            ))}
          </div>
          <div className="text-xs text-slate-500">
            Green = adds value, Red = reduces value. Segment width = influence.
          </div>
        </>
      )}
    </div>
  )
}
