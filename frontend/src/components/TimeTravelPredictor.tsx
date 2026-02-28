import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts'
import { Clock } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function TimeTravelPredictor({ dark, inputs }: Props) {
  const [years, setYears] = useState<{ year: number; multiplier: number; price: number }[]>([])
  const [currentPrice, setCurrentPrice] = useState<number | null>(null)

  const fetchTimeTravel = () => {
    fetch(`${API}/time-travel`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then(res => {
        if (!res.error) {
          setYears(res.years || [])
          setCurrentPrice(res.current_price)
        }
      })
  }

  useEffect(() => { fetchTimeTravel() }, [])

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white flex items-center gap-2">
        <Clock size={20} /> Time-Travel Predictor
      </h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        What would this house have cost in 2010? 2015? Historical estimates.
      </p>
      <button
        onClick={fetchTimeTravel}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
      >
        Travel Through Time
      </button>
      {currentPrice !== null && years.length > 0 && (
        <>
          <p className="text-lg font-bold text-indigo-600">Today: ${currentPrice.toLocaleString()}</p>
          <div className="h-48">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={years}>
                <XAxis dataKey="year" stroke={dark ? '#94a3b8' : '#64748b'} />
                <YAxis stroke={dark ? '#94a3b8' : '#64748b'} tickFormatter={v => `$${(v / 1000).toFixed(0)}k`} />
                <Tooltip formatter={(v: number) => [`$${v.toLocaleString()}`, 'Price']} />
                <Line type="monotone" dataKey="price" stroke="#6366f1" strokeWidth={2} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-2 gap-2">
            {years.map(y => (
              <div key={y.year} className="p-2 rounded bg-slate-100 dark:bg-slate-700 text-center">
                <p className="font-bold">{y.year}</p>
                <p className="text-indigo-600">${y.price.toLocaleString()}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}
