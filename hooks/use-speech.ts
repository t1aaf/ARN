"use client"

import { useState, useCallback, useRef, useEffect } from "react"

export function useSpeech() {
  const [isSpeaking, setIsSpeaking] = useState(false)
  const [isReady, setIsReady] = useState(false)
  const lastSpokenTextRef = useRef<string>("")
  const lastSpeakTimeRef = useRef<number>(0)
  const speakingTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  useEffect(() => {
    if (!("speechSynthesis" in window)) {
      console.error("[v0] Speech synthesis not supported")
      return
    }

    const loadVoices = () => {
      const voices = window.speechSynthesis.getVoices()
      if (voices.length > 0) {
        setIsReady(true)
        console.log("[v0] Speech synthesis ready with", voices.length, "voices")
      }
    }

    loadVoices()

    if (window.speechSynthesis.onvoiceschanged !== undefined) {
      window.speechSynthesis.onvoiceschanged = loadVoices
    }

    const timeout = setTimeout(() => setIsReady(true), 500)

    return () => {
      clearTimeout(timeout)
      window.speechSynthesis.cancel()
      if (speakingTimeoutRef.current) {
        clearTimeout(speakingTimeoutRef.current)
      }
    }
  }, [])

  const speak = useCallback(
    (text: string) => {
      if (!("speechSynthesis" in window) || !isReady) {
        console.log("[v0] Speech not ready")
        return
      }

      const now = Date.now()
      const timeSinceLastSpeak = now - lastSpeakTimeRef.current

      // Prevent speaking too frequently
      if (timeSinceLastSpeak < 2500) {
        console.log("[v0] Skipping speech - too soon")
        return
      }

      // Don't repeat the same text
      if (text === lastSpokenTextRef.current && timeSinceLastSpeak < 5000) {
        console.log("[v0] Skipping speech - same text")
        return
      }

      // Cancel any ongoing speech
      window.speechSynthesis.cancel()

      // Clear any existing timeout
      if (speakingTimeoutRef.current) {
        clearTimeout(speakingTimeoutRef.current)
      }

      console.log("[v0] Speaking:", text)
      setIsSpeaking(true)
      lastSpokenTextRef.current = text
      lastSpeakTimeRef.current = now

      const utterance = new SpeechSynthesisUtterance(text)
      utterance.rate = 1.0
      utterance.pitch = 1.0
      utterance.volume = 1.0
      utterance.lang = "en-US"

      utterance.onend = () => {
        console.log("[v0] Speech ended")
        setIsSpeaking(false)
      }

      utterance.onerror = (event) => {
        console.log("[v0] Speech error:", event)
        setIsSpeaking(false)
      }

      // Safety timeout in case onend doesn't fire
      speakingTimeoutRef.current = setTimeout(() => {
        console.log("[v0] Speech timeout - forcing end")
        setIsSpeaking(false)
      }, 5000)

      try {
        window.speechSynthesis.speak(utterance)
      } catch (error) {
        console.error("[v0] Failed to speak:", error)
        setIsSpeaking(false)
      }
    },
    [isReady],
  )

  const stopSpeaking = useCallback(() => {
    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel()
      setIsSpeaking(false)
      lastSpokenTextRef.current = ""
      if (speakingTimeoutRef.current) {
        clearTimeout(speakingTimeoutRef.current)
      }
    }
  }, [])

  const testSpeech = useCallback(() => {
    if (!("speechSynthesis" in window)) {
      console.error("[v0] Speech synthesis not supported")
      return false
    }

    try {
      const utterance = new SpeechSynthesisUtterance("Speech system ready")
      utterance.rate = 1.0
      utterance.volume = 1.0
      window.speechSynthesis.speak(utterance)
      console.log("[v0] Test speech triggered")
      return true
    } catch (error) {
      console.error("[v0] Test speech failed:", error)
      return false
    }
  }, [])

  return {
    speak,
    isSpeaking,
    stopSpeaking,
    isReady,
    testSpeech,
  }
}
