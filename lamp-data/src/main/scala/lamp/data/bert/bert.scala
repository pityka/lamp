package lamp.data.bert

import lamp.STen

case class BertData(
    maskedTokens: STen,
    segments: STen,
    predictionPositions: STen,
    maskedLanguageModelTarget: STen,
    nextSentenceTarget: STen
)
