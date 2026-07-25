import type { LoraTrainJobConfig } from '../defaults/loraTrain'

/** Serialize job (+ optional HF token) to Captioer trainer JSON. */
export function serializeTrainConfig(
  job: LoraTrainJobConfig,
  extras?: { huggingface_token?: string }
): string {
  const payload = {
    ...job,
    model: {
      ...job.model,
      name_or_path: job.model.train_name_or_path || job.model.name_or_path,
      train_name_or_path: job.model.train_name_or_path || job.model.name_or_path,
      arch: job.model.arch || 'krea2'
    },
    huggingface_token: extras?.huggingface_token || undefined
  }
  return `${JSON.stringify(payload, null, 2)}\n`
}
