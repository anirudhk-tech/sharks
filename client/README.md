# Setup Guide

## 1. Create `.env.local` file

Create a file named `.env.local` in the project root and paste this:

```env
# Mapbox
NEXT_PUBLIC_MAPBOX_ACCESS_TOKEN=pk.eyJ1Ijoid2F5cG9pbnR0cmFuc2l0IiwiYSI6ImNtYXJnbWU3cjBhajMyeW9nZnBpaWYxcW8ifQ.VlCcfOHVB6lSyIqbTxj8RQ

# AI Provider
OPENAI_API_KEY=your_openai_api_key_here

# Supabase (add these after step 3)
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=
```

## 2. Create Supabase Project

1. Go to [supabase.com](https://supabase.com) and sign in
2. Click **New Project**
3. Enter project name, password, and region
4. Wait ~1 minute for setup

## 3. Add Supabase Credentials

1. In Supabase dashboard: Click **Connect** at the top
2. Click **App Frameworks** and select **Next.js**
3. Copy the provided environment variables and add to `.env.local`:

## 4. Run SQL Setup

1. In Supabase dashboard: **SQL Editor** → **New query**
2. Copy all contents from `supabase-setup.sql`
3. Paste and click **Run**
4. Verify 5 rows inserted

## 5. Start Development Server

```bash
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000)


tool call: add pin to map to, add to supabase
position the map, tool call to shift position, map move to.
show me white house, .. ,  .... ,  stepOnWhen.