import { Sparkles } from 'lucide-react'
import type { PlaygroundInputs } from '../App'
import VoiceExplainer from './VoiceExplainer'
import StoryDecomposition from './StoryDecomposition'
import CounterfactualStories from './CounterfactualStories'
import SacrificeGame from './SacrificeGame'
import SensitivityTheater from './SensitivityTheater'
import DreamHomeDNA from './DreamHomeDNA'
import TimeTravelPredictor from './TimeTravelPredictor'
import ClimateRiskPanel from './ClimateRiskPanel'
import NeighborhoodTwins from './NeighborhoodTwins'
import ConfidenceLandscape from './ConfidenceLandscape'
import FairnessLens from './FairnessLens'
import SimpleExplain from './SimpleExplain'

type Props = { dark: boolean; inputs: PlaygroundInputs }

export default function ExperimentsHub({ dark, inputs }: Props) {
  const cards = [
    { title: 'AI Voice Explainer', C: VoiceExplainer },
    { title: 'Story Decomposition', C: StoryDecomposition },
    { title: 'Counterfactual Stories', C: CounterfactualStories },
    { title: 'What Would You Sacrifice?', C: SacrificeGame },
    { title: 'Sensitivity Theater', C: SensitivityTheater },
    { title: 'Dream Home DNA', C: DreamHomeDNA },
    { title: 'Time-Travel Predictor', C: TimeTravelPredictor },
    { title: 'Climate Risk', C: ClimateRiskPanel },
    { title: 'Neighborhood Twins', C: NeighborhoodTwins },
    { title: 'Confidence Landscape', C: ConfidenceLandscape },
    { title: 'Fairness Lens', C: FairnessLens },
    { title: 'Explainability for Everyone', C: SimpleExplain },
  ]

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-slate-800 dark:text-white flex items-center gap-2">
          <Sparkles size={28} /> Never Seen Before
        </h2>
        <p className="text-slate-600 dark:text-slate-400 mt-2">
          12 unique features that make SmartExplain AI stand out. Voice, stories, trade-offs, time travel, and more.
        </p>
        <p className="text-sm text-indigo-600 dark:text-indigo-400 mt-2 font-medium">
          All cards use the values from Prediction Playground. Set sliders there, then open any card and click Update to see results for that property.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6">
        {cards.map(({ title, C }) => (
          <div
            key={title}
            className="bg-white dark:bg-slate-800 rounded-xl p-6 shadow border border-slate-200 dark:border-slate-700"
          >
            <C dark={dark} inputs={inputs} />
          </div>
        ))}
      </div>
    </div>
  )
}
