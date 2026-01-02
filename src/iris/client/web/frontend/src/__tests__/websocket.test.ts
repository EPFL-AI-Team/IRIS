import { renderHook } from '@testing-library/react';
import { useBrowserStream } from '../hooks/useBrowserStream';
import { vi, describe, it, expect, beforeEach, afterEach } from 'vitest';

describe('useBrowserStream', () => {
  const mockSend = vi.fn();
  const mockClose = vi.fn();
  const connectSpy = vi.fn();

  beforeEach(() => {
    // Mock WebSocket
    class MockWebSocket {
      send = mockSend;
      close = mockClose;
      readyState = 1; // OPEN
      addEventListener = vi.fn();
      removeEventListener = vi.fn();
      onopen = null;
      onmessage = null;
      onerror = null;
      onclose = null;
      constructor(url: string) {
        connectSpy(url);
      }
    }

    vi.stubGlobal('WebSocket', MockWebSocket);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should be defined', () => {
    const { result } = renderHook(() => useBrowserStream());
    expect(result.current).toBeDefined();
    expect(result.current.connect).toBeDefined();
  });

  it('should instantiate WebSocket on connect', () => {
    const { result } = renderHook(() => useBrowserStream());
    
    result.current.connect();
    
    expect(connectSpy).toHaveBeenCalled();
  });
});
