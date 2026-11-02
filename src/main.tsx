import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { Shell } from './ui/Shell';
import './ui/styles.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Shell />
  </StrictMode>,
);
