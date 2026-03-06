import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, ResponsiveContainer, Tooltip } from 'recharts'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = {
  dark: boolean
  inputs: PlaygroundInputs
  onInputsChange: (inputs: PlaygroundInputs) => void
}

export default function PredictionPlayground({ dark, inputs, onInputsChange }: Props) {
  const [, setInfo] = useState<{ dataset?: string }>({})
  const [prediction, setPrediction] = useState<number | null>(null)
  const [contributions, setContributions] = useState<{ name: string; contribution: number; percent: number }[]>([])
  const [loading, setLoading] = useState(false)
  const [showBuild, setShowBuild] = useState(false)
  const [budget, setBudget] = useState<string>('')
  const setInputs = (next: PlaygroundInputs | ((prev: PlaygroundInputs) => PlaygroundInputs)) => {
    onInputsChange(typeof next === 'function' ? next(inputs) : next)
  }

  useEffect(() => {
    fetch(`${API}/info`).then(r => r.json()).then(setInfo)
  }, [])

  const predict = () => {
    const minDuration = 1600
    const startedAt = Date.now()
    setLoading(true)
    setShowBuild(true)
    fetch(`${API}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then((res) => {
        if (res.error) return
        setPrediction(res.prediction)
        setContributions(res.contributions || [])
      })
      .finally(() => {
        const elapsed = Date.now() - startedAt
        const remaining = minDuration - elapsed
        if (remaining > 0) {
          setTimeout(() => {
            setLoading(false)
            setShowBuild(false)
          }, remaining)
        } else {
          setLoading(false)
          setShowBuild(false)
        }
      })
  }

  return (
    <div className="space-y-6">
      <section className="space-y-4">
        <div className="flex flex-col gap-2">
          <h2 className="text-2xl md:text-3xl font-semibold text-slate-900 dark:text-slate-50">
            Find the fair price for your area
          </h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 max-w-2xl">
            Choose the neighbourhood, building profile, and income band. We keep the backend model exactly
            the same and show you a clean, portal-style price card.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,0.9fr)] gap-6">
          {/* Left: filters */}
          <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm">
            <div className="flex items-center justify-between px-5 pt-4 pb-3 border-b border-slate-100 dark:border-slate-800">
              <div>
                <p className="text-[0.7rem] uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">
                  Area & property filters
                </p>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Similar to “Locality / Budget / BHK” filters on property portals.
                </p>
              </div>
              <button
                onClick={() =>
                  setInputs({
                    ...inputs,
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
                }
                className="text-[0.7rem] px-3 py-1.5 rounded-full border border-slate-200 dark:border-slate-700 text-slate-600 dark:text-slate-200 hover:bg-slate-50 dark:hover:bg-slate-800"
              >
                Reset filters
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6 px-5 py-4">
              <div className="space-y-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Location & market
                </p>
                {[
                  { key: 'ocean_proximity', label: 'Ocean proximity', isSelect: true },
                  { key: 'median_income', label: 'Median income (x $10k)', min: 0.5, max: 15, step: 0.1 },
                  { key: 'housing_median_age', label: 'Housing median age (years)', min: 1, max: 52, step: 1 },
                  { key: 'population', label: 'Population in block', min: 3, max: 36000, step: 100 },
                ].map(({ key, label, min, max, step, isSelect }) => (
                  <div key={key}>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-xs font-medium text-slate-700 dark:text-slate-200">
                        {label}
                      </label>
                      {!isSelect && (
                        <span className="text-[0.7rem] text-slate-400">
                          {inputs[key as keyof typeof inputs] as number}
                        </span>
                      )}
                    </div>
                    {isSelect ? (
                      <select
                        value={inputs[key as keyof typeof inputs] as string}
                        onChange={(e) => setInputs({ ...inputs, [key]: e.target.value })}
                        className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800 text-sm"
                      >
                        {['INLAND', '<1H OCEAN', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND'].map((v) => (
                          <option key={v} value={v}>
                            {v}
                          </option>
                        ))}
                      </select>
                    ) : (
                      <input
                        type="range"
                        min={min}
                        max={max}
                        step={step}
                        value={inputs[key as keyof typeof inputs] as number}
                        onChange={(e) => setInputs({ ...inputs, [key]: Number(e.target.value) })}
                        className="w-full accent-indigo-600"
                      />
                    )}
                  </div>
                ))}
              </div>

              <div className="space-y-4">
                <p className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                  Building profile
                </p>
                {[
                  { key: 'total_rooms', label: 'Total rooms in block', min: 2, max: 40000, step: 100 },
                  { key: 'total_bedrooms', label: 'Total bedrooms', min: 1, max: 6500, step: 10 },
                  { key: 'households', label: 'Households', min: 1, max: 6100, step: 50 },
                  { key: 'longitude', label: 'Longitude', min: -124, max: -114, step: 0.1 },
                  { key: 'latitude', label: 'Latitude', min: 32, max: 42, step: 0.1 },
                ].map(({ key, label, min, max, step }) => (
                  <div key={key}>
                    <div className="flex items-center justify-between mb-1.5">
                      <label className="text-xs font-medium text-slate-700 dark:text-slate-200">
                        {label}
                      </label>
                      <span className="text-[0.7rem] text-slate-400">
                        {inputs[key as keyof typeof inputs] as number}
                      </span>
                    </div>
                    <input
                      type="range"
                      min={min}
                      max={max}
                      step={step}
                      value={inputs[key as keyof typeof inputs] as number}
                      onChange={(e) => setInputs({ ...inputs, [key]: Number(e.target.value) })}
                      className="w-full accent-indigo-600"
                    />
                  </div>
                ))}
              </div>
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 px-5 pb-4 pt-2 border-t border-slate-100 dark:border-slate-800">
              <p className="text-[0.7rem] text-slate-500 dark:text-slate-400">
                Tip: move from <span className="font-semibold">INLAND</span> to{' '}
                <span className="font-semibold">NEAR OCEAN</span> and watch the price band shift.
              </p>
              <button
                onClick={predict}
                disabled={loading}
                className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-700 text-white text-xs md:text-sm font-semibold shadow-sm disabled:opacity-60 disabled:cursor-not-allowed"
              >
                {loading ? 'Getting price…' : 'Get price estimate'}
              </button>
            </div>
          </div>

          {/* Right: price card */}
          <div className="space-y-4">
            <div className="bg-white dark:bg-slate-900 rounded-2xl border border-slate-200 dark:border-slate-800 shadow-sm p-5 md:p-6">
              <div className="flex items-center justify-between gap-3 mb-3">
                <div>
                  <p className="text-[0.7rem] uppercase tracking-[0.18em] text-slate-500 dark:text-slate-400">
                    Estimated price
                  </p>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Based on your selected area & building profile.
                  </p>
                </div>
                <div className="text-right">
                  <p className="text-[0.65rem] text-slate-400">Ocean proximity</p>
                  <p className="text-xs font-medium text-slate-700 dark:text-slate-200">
                    {inputs.ocean_proximity}
                  </p>
                </div>
              </div>

              {showBuild ? (
                <div className="py-10 flex flex-col items-center gap-4">
                  <div className="se-house-build">
                    <div className="se-house-ground" />
                    <div className="se-house-body se-house-part" />
                    <div className="se-house-roof se-house-part se-house-delay-1" />
                    <div className="se-house-window se-house-part se-house-delay-2" />
                    <div className="se-house-door se-house-part se-house-delay-3" />
                  </div>
                  <p className="text-xs text-slate-500 dark:text-slate-400">
                    Building your house profile…
                  </p>
                </div>
              ) : prediction !== null ? (
                <>
                  <div className="mb-4">
                    <p className="text-3xl md:text-4xl font-semibold tracking-tight text-slate-900 dark:text-slate-50">
                      ${prediction.toLocaleString()}
                    </p>
                    <p className="text-[0.7rem] text-slate-500 dark:text-slate-400 mt-1">
                      Model: high‑performance California housing regression (unchanged backend).
                    </p>
                  </div>

                  <div className="mb-4 p-3.5 rounded-xl bg-slate-50 dark:bg-slate-800/70 border border-slate-200 dark:border-slate-700">
                    <label className="block text-xs font-medium text-slate-600 dark:text-slate-300 mb-1.5">
                      Budget check
                    </label>
                    <input
                      type="number"
                      placeholder="Enter your budget (e.g. 200000)"
                      value={budget}
                      onChange={(e) => setBudget(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-slate-200 dark:border-slate-600 bg-white dark:bg-slate-900 text-sm"
                    />
                    {budget && Number(budget) > 0 && (
                      <p
                        className={`mt-2 text-xs font-semibold ${
                          prediction <= Number(budget)
                            ? 'text-emerald-600 dark:text-emerald-400'
                            : 'text-rose-600 dark:text-rose-400'
                        }`}
                      >
                        {prediction <= Number(budget)
                          ? `Under budget by $${(Number(budget) - prediction).toLocaleString()}`
                          : `Over budget by $${(prediction - Number(budget)).toLocaleString()}`}
                      </p>
                    )}
                  </div>

                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <h4 className="text-sm font-semibold text-slate-800 dark:text-slate-100">
                        Top factors for this estimate
                      </h4>
                      <p className="text-[0.7rem] text-slate-500 dark:text-slate-400">
                        Feature‑level impact
                      </p>
                    </div>
                    <div className="h-64">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={contributions.slice(0, 10)}
                          layout="vertical"
                          margin={{ left: 90, right: 8 }}
                        >
                          <XAxis type="number" stroke={dark ? '#94a3b8' : '#64748b'} />
                          <YAxis
                            type="category"
                            dataKey="name"
                            width={85}
                            stroke={dark ? '#94a3b8' : '#64748b'}
                          />
                          <Tooltip
                            contentStyle={{
                              background: dark ? '#020617' : '#ffffff',
                              border: '1px solid #334155',
                            }}
                          />
                          <Bar dataKey="contribution" fill="#6366f1" radius={[0, 4, 4, 0]} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </>
              ) : (
                <div className="py-10 text-center text-sm text-slate-500 dark:text-slate-400">
                  Adjust the filters on the left and click{' '}
                  <span className="font-semibold">Get price estimate</span> to see the valuation here.
                </div>
              )}
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}
