"use client"

import type React from "react"

import { useState, useEffect, useRef, useCallback } from "react"
import * as cocoSsd from "@tensorflow-models/coco-ssd"
import "@tensorflow/tfjs"

interface Detection {
  class: string
  score: number
  bbox: [number, number, number, number]
}

export function useObjectDetection(videoRef: React.RefObject<HTMLVideoElement>, isSpeaking = false) {
  const [detections, setDetections] = useState<string[]>([])
  const [isModelLoading, setIsModelLoading] = useState(true)
  const modelRef = useRef<cocoSsd.ObjectDetection | null>(null)
  const detectionIntervalRef = useRef<NodeJS.Timeout | null>(null)
  const lastAnnouncementRef = useRef<string>("")
  const lastAnnouncementTimeRef = useRef<number>(0)

  // Load model on mount
  useEffect(() => {
    const loadModel = async () => {
      try {
        console.log("[v0] Loading COCO-SSD model...")
        const model = await cocoSsd.load()
        modelRef.current = model
        setIsModelLoading(false)
        console.log("[v0] Model loaded successfully")
      } catch (error) {
        console.error("[v0] Error loading model:", error)
        setIsModelLoading(false)
      }
    }

    loadModel()

    return () => {
      if (detectionIntervalRef.current) {
        clearInterval(detectionIntervalRef.current)
      }
    }
  }, [])

  const estimateDistance = (bbox: [number, number, number, number], videoWidth: number): number => {
    const [, , width, height] = bbox
    const objectSize = Math.sqrt(width * height)
    const videoArea = videoWidth * (videoWidth * 0.75) // Assuming 4:3 aspect ratio
    const sizeRatio = objectSize / Math.sqrt(videoArea)

    // More precise distance estimation with decimal values
    if (sizeRatio > 0.35) return 1.2
    if (sizeRatio > 0.3) return 1.5
    if (sizeRatio > 0.25) return 1.8
    if (sizeRatio > 0.2) return 2.3
    if (sizeRatio > 0.15) return 3.1
    if (sizeRatio > 0.12) return 3.8
    if (sizeRatio > 0.1) return 4.5
    if (sizeRatio > 0.08) return 5.2
    if (sizeRatio > 0.06) return 6.1
    return 7.3
  }

  // Determine position (left, center, right)
  const getPosition = (bbox: [number, number, number, number], videoWidth: number): string => {
    const [x, , width] = bbox
    const centerX = x + width / 2
    const relativePosition = centerX / videoWidth

    if (relativePosition < 0.33) return "left"
    if (relativePosition > 0.67) return "right"
    return "center"
  }

  const detectWall = useCallback(
    (video: HTMLVideoElement): { hasWall: boolean; distance: number; position: string } | null => {
      const canvas = document.createElement("canvas")
      const ctx = canvas.getContext("2d")
      if (!ctx) return null

      canvas.width = video.videoWidth
      canvas.height = video.videoHeight
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      // Divide frame into 3 sections (left, center, right)
      const sectionWidth = canvas.width / 3
      const sections = ["left", "center", "right"]
      const results: Array<{ position: string; brightness: number; uniformity: number }> = []

      for (let i = 0; i < 3; i++) {
        const x = i * sectionWidth
        const imageData = ctx.getImageData(x, canvas.height / 3, sectionWidth, canvas.height / 3)
        const data = imageData.data

        let totalBrightness = 0
        let pixelCount = 0

        // Calculate average brightness
        for (let j = 0; j < data.length; j += 4) {
          const r = data[j]
          const g = data[j + 1]
          const b = data[j + 2]
          const brightness = (r + g + b) / 3
          totalBrightness += brightness
          pixelCount++
        }

        const avgBrightness = totalBrightness / pixelCount

        // Calculate uniformity (low variance = uniform surface like wall)
        let variance = 0
        for (let j = 0; j < data.length; j += 4) {
          const r = data[j]
          const g = data[j + 1]
          const b = data[j + 2]
          const brightness = (r + g + b) / 3
          variance += Math.pow(brightness - avgBrightness, 2)
        }
        const uniformity = 1 - Math.sqrt(variance / pixelCount) / 255

        results.push({
          position: sections[i],
          brightness: avgBrightness,
          uniformity,
        })
      }

      // Find section with highest uniformity (likely a wall)
      const mostUniform = results.reduce((max, current) => (current.uniformity > max.uniformity ? current : max))

      // If uniformity is high enough, it's likely a wall
      if (mostUniform.uniformity > 0.7) {
        let distance = 5.5
        if (mostUniform.brightness > 200) distance = 1.5
        else if (mostUniform.brightness > 180) distance = 2.2
        else if (mostUniform.brightness > 160) distance = 2.8
        else if (mostUniform.brightness > 140) distance = 3.4
        else if (mostUniform.brightness > 120) distance = 4.1
        else if (mostUniform.brightness > 100) distance = 4.7

        return {
          hasWall: true,
          distance,
          position: mostUniform.position,
        }
      }

      return null
    },
    [],
  )

  // Detect objects in video frame
  const detectObjects = useCallback(async () => {
    if (isSpeaking) {
      return
    }

    const video = videoRef.current
    const model = modelRef.current

    if (!video || !model || video.readyState !== 4) return

    try {
      const predictions = await model.detect(video)

      // Filter for people first
      const people = predictions.filter((p) => p.class === "person")

      if (people.length > 0) {
        // Process person detections
        const closestPerson = people.reduce((closest, current) => {
          const closestDist = estimateDistance(closest.bbox, video.videoWidth)
          const currentDist = estimateDistance(current.bbox, video.videoWidth)
          return currentDist < closestDist ? current : closest
        })

        const distance = estimateDistance(closestPerson.bbox, video.videoWidth)
        const position = getPosition(closestPerson.bbox, video.videoWidth)

        let announcement = ""
        if (distance < 2) {
          announcement = `Very close person ${position} about ${distance} meters`
        } else {
          announcement = `Person ${position} about ${distance} meters away`
        }

        // Only announce if different from last announcement or enough time has passed
        const now = Date.now()
        if (announcement !== lastAnnouncementRef.current || now - lastAnnouncementTimeRef.current > 3000) {
          setDetections([announcement])
          lastAnnouncementRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
        return
      }

      // Check for other obstacles (furniture, etc.)
      const obstacles = predictions.filter((p) =>
        ["chair", "couch", "bed", "dining table", "door", "tv", "laptop", "bottle", "cup"].includes(p.class),
      )

      if (obstacles.length > 0) {
        const closestObstacle = obstacles.reduce((closest, current) => {
          const closestDist = estimateDistance(closest.bbox, video.videoWidth)
          const currentDist = estimateDistance(current.bbox, video.videoWidth)
          return currentDist < closestDist ? current : closest
        })

        const distance = estimateDistance(closestObstacle.bbox, video.videoWidth)
        const position = getPosition(closestObstacle.bbox, video.videoWidth)

        const announcement = `${closestObstacle.class} ${position} about ${distance} meters away`

        const now = Date.now()
        if (announcement !== lastAnnouncementRef.current || now - lastAnnouncementTimeRef.current > 3000) {
          setDetections([announcement])
          lastAnnouncementRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
        return
      }

      const wallDetection = detectWall(video)
      if (wallDetection && wallDetection.hasWall) {
        const announcement = `Wall ${wallDetection.position} about ${wallDetection.distance} meters ahead`

        const now = Date.now()
        if (announcement !== lastAnnouncementRef.current || now - lastAnnouncementTimeRef.current > 3000) {
          setDetections([announcement])
          lastAnnouncementRef.current = announcement
          lastAnnouncementTimeRef.current = now
        }
        return
      }

      // No objects or walls detected
      const now = Date.now()
      if (now - lastAnnouncementTimeRef.current > 5000) {
        setDetections(["Clear path ahead"])
        lastAnnouncementRef.current = "Clear path ahead"
        lastAnnouncementTimeRef.current = now
      }
    } catch (error) {
      console.error("[v0] Detection error:", error)
    }
  }, [videoRef, isSpeaking, detectWall])

  // Start detection loop
  const startDetection = useCallback(async () => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
    }

    detectionIntervalRef.current = setInterval(() => {
      detectObjects()
    }, 800)
  }, [detectObjects])

  // Stop detection loop
  const stopDetection = useCallback(() => {
    if (detectionIntervalRef.current) {
      clearInterval(detectionIntervalRef.current)
      detectionIntervalRef.current = null
    }
    setDetections([])
    lastAnnouncementRef.current = ""
  }, [])

  return {
    detections,
    isModelLoading,
    startDetection,
    stopDetection,
  }
}
