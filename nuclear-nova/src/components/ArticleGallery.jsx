import { createRoot } from 'react-dom/client';
import { Lightbox } from '@astryxdesign/core/Lightbox';

export function openGallery(host, media, index, trigger) {
  const root = createRoot(host);
  const close = () => {
    root.unmount();
    host.remove();
    if (trigger.isConnected) trigger.focus({ preventScroll: true });
  };
  root.render(
    <Lightbox
      isOpen
      defaultIndex={index}
      media={media}
      hasZoom
      onOpenChange={(isOpen) => { if (!isOpen) close(); }}
    />
  );
  return close;
}
