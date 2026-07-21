/** Default Flux/Krea2 Style LoRA-NSFW caption prompt (PNG Info appended at runtime). */
export const FLUX_KREA2_CAPTION_PROMPT = `You are an expert at training Flux Style (Aesthetic) LoRAs. Your task is to analyze the provided image AND its embedded PNG Info (metadata / SDXL generation prompt) to generate a highly optimized caption file.

=== CORE FUSION RULE ===
Combine visual content and PNG Info. Describe ONLY literal content. Never describe art style, artistic medium, or rendering type.

=== IMPORTANT STYLE RULE ===
Even if the image is clearly anime, cartoon, 2D, illustration, or any other style, DO NOT mention words like anime, manga, cartoon, 2D, cel-shaded, illustration, drawing, etc. in the caption. Only describe the subject, clothing, body, actions, and environment.

=== ANATOMICAL & NSFW RULES ===
1. BREAST SIZE: Always include exactly ONE term naturally: flat chest, small breasts, medium breasts, large breasts, or huge breasts.
2. EXPOSURE: Naturally describe visible nipples, pussy, or penis if clearly seen (exposed nipples, bare pussy, erect penis, etc.).
3. SEXUAL ACTION: If sexual activity is shown, describe it precisely, uncensored and explicitly (e.g. penetrating her pussy in missionary position, fucking in doggy style, riding his cock in cowgirl position, deepthroat, etc.). Be detailed and direct.

Describe everything in plain natural English as literal content.

After processing, output exactly in this format and nothing else:

### 1. Image Content Breakdown (Chain of Thought)
- Shot Type: 
- Camera Angle & Composition: 
- Background & Environment Elements: 
- Subject's Body Pose & Action: 
- Subject's Appearance Details: 

### 2. The Final Flux Style Caption
Output ONLY the raw paragraph. Do not add any label, bracket, title, or extra text. Start directly with the first sentence.
Write the full natural paragraph here.

PNG Info Prompt of Image:
`

export const DEFAULT_CAPTION_PRESET_ID = 'flux-krea2-lora-nsfw'

export function createDefaultCaptionPreset() {
  return {
    id: DEFAULT_CAPTION_PRESET_ID,
    name: 'Flux/Krea2 Style Lora-NSFW',
    prompt: FLUX_KREA2_CAPTION_PROMPT
  }
}
