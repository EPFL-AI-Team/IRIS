/**
 * Format inference results as natural language sentences.
 * Mirrors the logic from src/iris/dataset/training_format.py
 */

const VERB_PREPOSITIONS: Record<string, string> = {
  insert: "into",
  put: "on",
  take: "from",
  press: "on",
  release: "of",
  detach: "from",
  open: "of",
  close: "of",
  eject: "into",
  shake: "of",
};

/**
 * Format a result object as a natural language sentence.
 * Example: "The left hand is inserting the pipette into the tube."
 *
 * @param result - Parsed JSON result from inference
 * @returns Natural language description
 */
export function formatResultAsNaturalLanguage(result: Record<string, unknown>): string {
  // Extract fields (handle both "verb"/"action" and "hand"/"hand_side" variations)
  const hand = String(result.hand || result.hand_side || "unknown");
  const verb = String(result.verb || result.action || "unknown");
  const tool = String(result.tool || "unknown").replace(/_/g, " ");
  const target = result.target;

  // Case 1: No target object
  if (!target || target === "none" || target === "null" || target === "nan") {
    return `The ${hand} hand is performing a '${verb}' action using the ${tool}.`;
  }

  // Case 2: With target object
  const targetStr = String(target).replace(/_/g, " ");
  const prep = VERB_PREPOSITIONS[verb] || "with";

  // Convert verb to present continuous (-ing form)
  const verbIng = verb.endsWith("e") ? verb.slice(0, -1) + "ing" : verb + "ing";

  return `The ${hand} hand is ${verbIng} the ${tool} ${prep} the ${targetStr}.`;
}

/**
 * Check if a string is valid JSON.
 */
export function isValidJSON(str: string): boolean {
  try {
    JSON.parse(str);
    return true;
  } catch {
    return false;
  }
}

/**
 * Parse and format a result string.
 * If it's valid JSON, parse it and format as natural language.
 * Otherwise, return the raw string.
 */
export function parseAndFormatResult(resultString: string): { formatted: string; parsed: Record<string, unknown> | null } {
  if (!isValidJSON(resultString)) {
    return { formatted: resultString, parsed: null };
  }

  try {
    const parsed = JSON.parse(resultString) as Record<string, unknown>;
    const formatted = formatResultAsNaturalLanguage(parsed);
    return { formatted, parsed };
  } catch {
    return { formatted: resultString, parsed: null };
  }
}
