import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip, Cell } from 'recharts'

const API = '/api'

export default function ExplainabilityHub({ dark }: { dark: boolean }) {
  const [prediction, setPrediction] = useState<number | null>(null)
  const [contributions, setContributions] = useState<{ name: string; contribution: number; percent: number }[]>([])
  const [inputs, setInputs] = useState({
    lot_area: 8450,
    overall_qual: 7,
    gr_liv_area: 1710,
    garage_cars: 2,
    total_bsmt_sf: 856,
    year_built: 2003,
    full_bath: 2,
    fireplace: 0,
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

  return (
    <div className="space-y-8">
      <h2 className="text-2xl font-bold text-slate-800 dark:text-white">Explainability Hub</h2>
      <p className="text-slate-600 dark:text-slate-400">
        See how each feature contributes to the predicted price. Green = increases, Red = decreases.
      </p>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1 bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold mb-4 text-slate-800 dark:text-white">Quick Inputs</h3>
          <div className="space-y-3">
            {[
              { key: 'overall_qual', label: 'Overall Quality', min: 1, max: 10 },
              { key: 'gr_liv_area', label: 'Living Area', min: 300, max: 6000 },
              { key: 'lot_area', label: 'Lot Area', min: 1000, max: 50000 },
            ].map(({ key, label, min, max }) => (
              <div key={key}>
                <label className="text-sm text-slate-600 dark:text-slate-400">{label}</label>
                <input
                  type="range"
                  min={min}
                  max={max}
                  value={inputs[key as keyof typeof inputs]}
                  onChange={(e) => setInputs({ ...inputs, [key]: Number(e.target.value) })}
                  className="w-full"
                />
              </div>
            ))}
            <button
              onClick={fetchExplain}
              className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
            >
              Update
            </button>
          </div>
        </div>

        <div className="lg:col-span-2 bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700">
          <h3 className="font-semibold mb-4 text-slate-800 dark:text-white">Waterfall: Contribution to Price</h3>
          {prediction !== null && (
            <>
              <p className="text-2xl font-bold text-indigo-600 mb-4">${prediction.toLocaleString()}</p>
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
