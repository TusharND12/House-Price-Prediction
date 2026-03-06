import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts'

const API = '/api'

export default function ExplainabilityHub({ dark }: { dark: boolean }) {
  const [prediction, setPrediction] = useState<number | null>(null)
  const [contributions, setContributions] = useState<{ name: string; contribution: number; percent: number }[]>([])
  const [inputs, setInputs] = useState({
    total_rooms: 2635,
    total_bedrooms: 537,
    housing_median_age: 29,
    median_income: 3.87,
    population: 1425,
    households: 499,
    longitude: -119,
    latitude: 36,
    ocean_proximity: 'INLAND',
  })

  const fetchExplain = () => {
    fetch(API + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then((r) => r.json())
      .then((res) => {
        if (!res.error) {
          setPrediction(res.prediction)
          setContributions(res.contributions || [])
        }
      })
  }

  useEffect(() => {
    fetchExplain()
  }, [])

  const waterfallData = contributions.slice(0, 8).map((c) => ({
    name: c.name.replace(/_/g, ' '),
    value: c.contribution,
    fill: c.contribution >= 0 ? '#22c55e' : '#ef4444',
  }))

  const hasData = prediction !== null && contributions.length > 0
  const totalAbs = contributions.reduce((sum, c) => sum + Math.abs(c.contribution), 0)
  const top3Abs = contributions.slice(0, 3).reduce((sum, c) => sum + Math.abs(c.contribution), 0)
  const top3Share = totalAbs > 0 ? (top3Abs / totalAbs) * 100 : 0
  const topPositive = [...contributions].filter((c) => c.contribution > 0).sort((a, b) => b.contribution - a.contribution)[0]
  const topNegative = [...contributions].filter((c) => c.contribution < 0).sort((a, b) => a.contribution - b.contribution)[0]

  return (
    <div className="space-y-8">
      <div className="space-y-1">
        <h2 className="text-2xl font-bold text-slate-800 dark:text-white">Explainability Hub</h2>
        <p className="text-slate-600 dark:text-slate-400 text-sm max-w-3xl">
          Understand exactly <span className="font-semibold">why</span> this price was predicted. Green bars push the price up,
          red bars pull it down. Use the sliders to see how the story changes for different neighbourhoods.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700 space-y-4">
          <div>
            <h3 className="font-semibold mb-1 text-slate-800 dark:text-white">Quick Inputs</h3>
            <p className="text-xs text-slate-500 dark:text-slate-400">
              Roughly set the kind of block you are interested in. We keep the backend model exactly the same.
            </p>
          </div>
          <div className="space-y-3">
            {[
              { key: 'total_rooms', label: 'Total Rooms', min: 2, max: 15000, step: 100 },
              { key: 'median_income', label: 'Median Income', min: 0.5, max: 15, step: 0.1 },
              { key: 'housing_median_age', label: 'Housing Age', min: 1, max: 52, step: 1 },
            ].map(({ key, label, min, max, step = 1 }) => (
              <div key={key}>
                <label className="text-sm text-slate-600 dark:text-slate-400">{label}</label>
                <input
                  type="range"
                  min={min}
                  max={max}
                  step={step}
                  value={inputs[key as keyof typeof inputs] as number}
                  onChange={(e) => setInputs({ ...inputs, [key]: Number(e.target.value) })}
                  className="w-full"
                />
              </div>
            ))}
            <button
              onClick={fetchExplain}
              className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg text-sm font-medium"
            >
              Update
            </button>
          </div>
        </div>

        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700 space-y-4">
          <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
            <div>
              <h3 className="font-semibold text-slate-800 dark:text-white">Waterfall: Contribution to Price</h3>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Each bar shows how much that feature adds to or subtracts from the base price for this block.
              </p>
            </div>
            {hasData && (
              <div className="flex flex-wrap gap-2 text-xs">
                <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300 border border-emerald-100 dark:border-emerald-800">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  Top 3 explain ~{top3Share.toFixed(0)}% of impact
                </span>
                {topPositive && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-emerald-50 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-300 border border-emerald-100 dark:border-emerald-800">
                    Biggest booster: {topPositive.name.replace(/_/g, ' ')}
                  </span>
                )}
                {topNegative && (
                  <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full bg-rose-50 text-rose-700 dark:bg-rose-900/30 dark:text-rose-300 border border-rose-100 dark:border-rose-800">
                    Biggest reducer: {topNegative.name.replace(/_/g, ' ')}
                  </span>
                )}
              </div>
            )}
          </div>
          {hasData && (
            <>
              <div className="flex items-baseline justify-between gap-4">
                <div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">Predicted price for this block</p>
                  <p className="text-2xl md:text-3xl font-bold text-indigo-600">
                    ${prediction!.toLocaleString()}
                  </p>
                </div>
                <div className="text-right text-xs text-slate-500 dark:text-slate-400">
                  <p>
                    ocean: <span className="font-medium">{inputs.ocean_proximity}</span>
                  </p>
                  <p>
                    rooms: <span className="font-medium">{inputs.total_rooms.toLocaleString()}</span>, income:{' '}
                    <span className="font-medium">{inputs.median_income.toFixed(1)}x</span>
                  </p>
                </div>
              </div>
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={waterfallData} layout="vertical" margin={{ left: 100 }}>
                    <XAxis type="number" stroke={dark ? '#94a3b8' : '#64748b'} />
                    <YAxis type="category" dataKey="name" width={90} stroke={dark ? '#94a3b8' : '#64748b'} />
                    <Tooltip contentStyle={{ background: dark ? '#1e293b' : '#fff', border: '1px solid #334155' }} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                      {waterfallData.map((entry, i) => (
                        <Cell key={i} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
