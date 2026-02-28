import { useState, useRef } from 'react'
import { Volume2, VolumeX } from 'lucide-react'
import type { PlaygroundInputs } from '../App'

const API = '/api'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function VoiceExplainer({ dark, inputs }: Props) {
  const [text, setText] = useState('')
  const [prediction, setPrediction] = useState<number | null>(null)
  const [speaking, setSpeaking] = useState(false)
  const synthRef = useRef<SpeechSynthesis | null>(null)

  const explain = () => {
    fetch(`${API}/voice-explain`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(inputs),
    })
      .then(r => r.json())
      .then(res => {
        if (!res.error) {
          setText(res.text)
          setPrediction(res.prediction)
        }
      })
  }

  const speak = () => {
    if (!text || !window.speechSynthesis) return
    if (speaking) {
      window.speechSynthesis.cancel()
      setSpeaking(false)
      return
    }
    const u = new SpeechSynthesisUtterance(text)
    u.rate = 0.9
    u.onend = () => setSpeaking(false)
    window.speechSynthesis.speak(u)
    setSpeaking(true)
  }

  return (
    <div className="space-y-6">
      <h3 className="font-semibold text-slate-800 dark:text-white">AI Voice Explainer</h3>
      <p className="text-sm text-slate-600 dark:text-slate-400">
        Hear the model explain the price in plain English—using values from Prediction Playground.
      </p>
      <p className="text-xs text-slate-500">Living area: {inputs.gr_liv_area} sq ft · Quality: {inputs.overall_qual}</p>
      <button
        onClick={explain}
        className="w-full py-2 bg-indigo-600 hover:bg-indigo-700 text-white rounded-lg"
      >
        Generate Explanation
      </button>
      {text && (
        <div className="p-4 bg-slate-100 dark:bg-slate-700 rounded-lg">
          {prediction && (
            <p className="text-lg font-bold text-indigo-600 dark:text-indigo-400 mb-2">
              ${prediction.toLocaleString()}
            </p>
          )}
          <p className="text-slate-700 dark:text-slate-200 mb-3">{text}</p>
          <button
            onClick={speak}
            className={`flex items-center gap-2 px-3 py-2 rounded-lg ${speaking ? 'bg-amber-600' : 'bg-emerald-600 hover:bg-emerald-700'} text-white`}
          >
            {speaking ? <VolumeX size={18} /> : <Volume2 size={18} />}
            {speaking ? 'Stop' : 'Listen'}
          </button>
        </div>
      )}
    </div>
  )
}
